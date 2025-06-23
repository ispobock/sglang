#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <torch/all.h>

template <typename T, int VEC_SIZE>
__global__ void
moe_sum_reduce_kernel(const T* __restrict__ input, T* __restrict__ output, int M, int TOP_K, int DIM, float scaling) {
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
    acc[i] *= scaling;
  }

  Vec* out_vec = reinterpret_cast<Vec*>(output + m * DIM);
  Vec out;

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    out[i] = ToElem::convert(acc[i]);
  }

  out_vec[vec_idx] = out;
}

template <typename T, int VEC_SIZE>
void launch_moe_sum_reduce_kernel(const T* in, T* out, int M, int TOP_K, int DIM, float scaling, cudaStream_t stream) {
  constexpr int THREADS = 128;
  dim3 block(THREADS);
  dim3 grid(M, (DIM / VEC_SIZE + THREADS - 1) / THREADS);

  moe_sum_reduce_kernel<T, VEC_SIZE><<<grid, block, 0, stream>>>(in, out, M, TOP_K, DIM, scaling);
}

template <typename T>
void dispatch_type(torch::Tensor& in, torch::Tensor& out, float scaling) {
  int M = in.size(0);
  int TOP_K = in.size(1);
  int DIM = in.size(2);
  static constexpr int VEC_SIZE = 4;

  const T* in_ptr = in.data_ptr<T>();
  T* out_ptr = out.data_ptr<T>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  launch_moe_sum_reduce_kernel<T, VEC_SIZE>(in_ptr, out_ptr, M, TOP_K, DIM, scaling, stream);
}

void moe_sum_reduce_launcher(torch::Tensor input, torch::Tensor output, float scaling) {
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

  if (input.scalar_type() == at::ScalarType::Half) {
    dispatch_type<cutlass::half_t>(input, output, scaling);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_type<cutlass::bfloat16_t>(input, output, scaling);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype");
  }
}
