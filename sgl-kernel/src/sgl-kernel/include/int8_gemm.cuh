#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include "cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/gemm/gemm_universal_base_compat.h"
#include "cutlass_extensions/gemm/gemm_with_epilogue_visitor.h"
#include "utils.h"

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

void sm90_int8_scaled_mm(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                         const torch::Tensor& scales_a, const torch::Tensor& scales_b, const torch::Dtype& out_dtype,
                         const c10::optional<torch::Tensor>& bias);
#endif
