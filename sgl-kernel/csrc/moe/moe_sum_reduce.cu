#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <torch/all.h>

#include "utils.h"

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

template <typename T, int VEC_SIZE = 4>
__global__ void moe_sum_reduce_warp_kernel(
    const T* __restrict__ input, T* __restrict__ output, int M, int TOP_K, int DIM, float scale_factor) {
  using Vec = cutlass::AlignedArray<T, VEC_SIZE>;
  using ToFloat = cutlass::NumericConverter<float, T>;
  using ToElem = cutlass::NumericConverter<T, float>;

  int m = blockIdx.x;

  int lane_id = threadIdx.x & (WARP_SIZE - 1);
  int warp_id = threadIdx.x / WARP_SIZE;

  int vec_idx = (blockIdx.y * (blockDim.x / WARP_SIZE) + warp_id) * VEC_SIZE;

  if (vec_idx + VEC_SIZE > DIM) {
    return;
  }

  float acc[VEC_SIZE] = {0.f};

  if (lane_id < TOP_K) {
    int k = lane_id;

    const Vec* in_vec = reinterpret_cast<const Vec*>(input + ((m * TOP_K + k) * DIM));
    Vec v = in_vec[vec_idx];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] += ToFloat::convert(v[i]);
    }
  }

#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] += SGLANG_SHFL_XOR_SYNC_WIDTH(0xffffffff, acc[i], offset, WARP_SIZE);
    }
  }

  if (lane_id == 0) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      acc[i] *= scale_factor;
    }

    Vec* out_vec = reinterpret_cast<Vec*>(output + m * DIM);
    Vec out;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      out[i] = ToElem::convert(acc[i]);
    }
    out_vec[vec_idx] = out;
  }
}

template <typename T, int VEC_SIZE>
void launch_moe_sum_reduce_kernel(const T* in, T* out, size_t M, size_t TOP_K, size_t DIM, float scaling_factor) {
  //   constexpr int THREADS = 128;
  //   dim3 block(THREADS);
  //   dim3 grid(M, (DIM / VEC_SIZE + THREADS - 1) / THREADS);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  //   moe_sum_reduce_kernel<T, VEC_SIZE><<<grid, block, 0, stream>>>(in, out, M, TOP_K, DIM, scaling_factor);

  constexpr int THREADS_PER_BLOCK = 128;
  int warp_count_per_block = THREADS_PER_BLOCK / WARP_SIZE;

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(M, (DIM / VEC_SIZE + warp_count_per_block - 1) / warp_count_per_block);

  moe_sum_reduce_warp_kernel<T, VEC_SIZE><<<grid, block, 0, stream>>>(in, out, M, TOP_K, DIM, scaling_factor);
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
