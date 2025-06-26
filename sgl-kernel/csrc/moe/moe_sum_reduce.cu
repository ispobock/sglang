#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <torch/all.h>

template <typename T, int VEC_SIZE>
__global__ void moe_sum_reduce_kernel(
    const T* __restrict__ input, T* __restrict__ output, size_t M, size_t TOP_K, size_t DIM, float scaling_factor) {
  using Vec = cutlass::AlignedArray<T, VEC_SIZE>;
  using ToFloat = cutlass::NumericConverter<float, T>;
  using ToElem = cutlass::NumericConverter<T, float>;

  int m = blockIdx.x;
  int vec_idx = blockIdx.y * blockDim.x + threadIdx.x;
  int hidden_id = vec_idx * VEC_SIZE;

  if (m >= M || hidden_id >= DIM) return;

  float acc[VEC_SIZE] = {0.f};

  for (int k = 0; k < TOP_K; ++k) {
    const Vec* in_vec = reinterpret_cast<const Vec*>(input + ((m * TOP_K + k) * DIM));
    Vec v = in_vec[vec_idx];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] += ToFloat::convert(v[i]);
    }
  }

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    acc[i] *= scaling_factor;
  }

  Vec* out_vec = reinterpret_cast<Vec*>(output + m * DIM);
  Vec out;

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    out[i] = ToElem::convert(acc[i]);
  }

  out_vec[vec_idx] = out;
}

template <typename T, int VEC_SIZE = 8, int CHUNK_VEC = 128>
__global__ void moe_sum_reduce_kernel_cp_async(
    const T* __restrict__ input,  // [M, TOP_K, DIM]
    T* __restrict__ output,       // [M, DIM]
    int M,
    int TOP_K,
    int DIM,
    float scaling) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using Vec = cutlass::AlignedArray<T, VEC_SIZE>;
  using ToFloat = cutlass::NumericConverter<float, T>;
  using ToElem = cutlass::NumericConverter<T, float>;

  const int m = blockIdx.x;
  const int tid = threadIdx.x;
  const int vecs_per_tok = DIM / VEC_SIZE;

  // double buffer
  extern __shared__ __align__(16) char smem_raw[];
  Vec* sm_stage0 = reinterpret_cast<Vec*>(smem_raw);
  Vec* sm_stage1 = sm_stage0 + TOP_K * CHUNK_VEC;
  Vec* sm_buf[2] = {sm_stage0, sm_stage1};

  int stage = 0;

  // prefetch
  if (tid < CHUNK_VEC && tid < vecs_per_tok) {
    for (int k = 0; k < TOP_K; ++k) {
      const Vec* gptr = reinterpret_cast<const Vec*>(input + ((m * TOP_K + k) * DIM));
      const void* gmem_ptr = &gptr[tid];
      uint32_t smem_u32 = __cvta_generic_to_shared(&sm_buf[0][k * CHUNK_VEC + tid]);
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_u32), "l"(gmem_ptr), "n"(sizeof(Vec)));
    }
  }
  asm volatile("cp.async.commit_group;\n");
  asm volatile("cp.async.wait_group 0;\n");
  __syncthreads();

  // main loop
  for (int base = 0; base < vecs_per_tok; base += CHUNK_VEC) {
    // prefetch next chunk
    int next_base = base + CHUNK_VEC;
    int remain_vecs_copy = vecs_per_tok - next_base;
    int vec2copy = remain_vecs_copy > 0 ? min(remain_vecs_copy, CHUNK_VEC) : 0;
    if (vec2copy && tid < vec2copy) {
      for (int k = 0; k < TOP_K; ++k) {
        const Vec* gptr = reinterpret_cast<const Vec*>(input + ((m * TOP_K + k) * DIM)) + next_base;
        const void* gmem_ptr = &gptr[tid];
        uint32_t smem_u32 = __cvta_generic_to_shared(&sm_buf[stage ^ 1][k * CHUNK_VEC + tid]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_u32), "l"(gmem_ptr), "n"(sizeof(Vec)));
      }
      asm volatile("cp.async.commit_group;\n");
    }

    // compute + write back
    int remain_vecs_compute = vecs_per_tok - base;
    int vec2compute = remain_vecs_compute > 0 ? min(remain_vecs_compute, CHUNK_VEC) : 0;
    if (vec2compute && tid < vec2compute) {
      float acc[VEC_SIZE] = {0.f};
      for (int k = 0; k < TOP_K; ++k) {
        Vec v = sm_buf[stage][k * CHUNK_VEC + tid];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += ToFloat::convert(v[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        acc[i] *= scaling;
      }

      Vec out;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        out[i] = ToElem::convert(acc[i]);
      }

      reinterpret_cast<Vec*>(output + m * DIM)[base + tid] = out;
    }
    __syncthreads();

    // wait for next chunk
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    stage ^= 1;
  }
#endif
}

template <typename T, int VEC_SIZE>
void launch_moe_sum_reduce_kernel(const T* in, T* out, size_t M, size_t TOP_K, size_t DIM, float scaling_factor) {
  // constexpr int THREADS = 128;
  // dim3 block(THREADS);
  // dim3 grid(M, (DIM / VEC_SIZE + THREADS - 1) / THREADS);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // moe_sum_reduce_kernel<T, VEC_SIZE><<<grid, block, 0, stream>>>(in, out, M, TOP_K, DIM, scaling_factor);

  constexpr int CHUNK_VEC = 128;
  dim3 block(CHUNK_VEC);
  size_t smem = 2 * CHUNK_VEC * TOP_K * VEC_SIZE * sizeof(T);
  int max_shmem = 227 * 1024;
  cudaFuncSetAttribute(
      moe_sum_reduce_kernel_cp_async<T, VEC_SIZE, CHUNK_VEC>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem);
  moe_sum_reduce_kernel_cp_async<T, VEC_SIZE, CHUNK_VEC>
      <<<M, block, smem, stream>>>(in, out, M, TOP_K, DIM, scaling_factor);
}

void moe_sum_reduce(const torch::Tensor& input, torch::Tensor& output, double scaling_factor) {
  TORCH_CHECK(input.device().is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16,
      "Only FP16/BF16 input supported");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtypes must match");

  TORCH_CHECK(input.dim() == 3 && output.dim() == 2, "Bad shape");
  TORCH_CHECK(input.size(0) == output.size(0), "M mismatch");
  TORCH_CHECK(input.size(2) == output.size(1), "DIM mismatch");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Need contiguous");

  size_t M = input.size(0);
  size_t TOP_K = input.size(1);
  size_t DIM = input.size(2);
  static constexpr int VEC_SIZE = 8;

  if (input.scalar_type() == at::ScalarType::Half) {
    launch_moe_sum_reduce_kernel<cutlass::half_t, VEC_SIZE>(
        reinterpret_cast<const cutlass::half_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t*>(output.data_ptr<at::Half>()),
        M,
        TOP_K,
        DIM,
        static_cast<float>(scaling_factor));
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    launch_moe_sum_reduce_kernel<cutlass::bfloat16_t, VEC_SIZE>(
        reinterpret_cast<const cutlass::bfloat16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr<at::BFloat16>()),
        M,
        TOP_K,
        DIM,
        static_cast<float>(scaling_factor));
  } else {
    TORCH_CHECK(false, "Unsupported input dtype");
  }
}
