#include <cuda.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12000

#include "int8_gemm.cuh"

// #include <ATen/cuda/CUDAContext.h>
// #include <cutlass/cutlass.h>
// #include <cutlass/gemm/device/gemm.h>
// #include <cutlass/gemm/device/gemm_universal_adapter.h>
// #include <cutlass/numeric_types.h>

// #include <cute/atom/mma_atom.hpp>
// #include <cute/tensor.hpp>
// #include <cutlass/epilogue/collective/collective_builder.hpp>
// #include <cutlass/gemm/collective/collective_builder.hpp>
// #include <cutlass/gemm/kernel/gemm_universal.hpp>
// #include <cutlass/util/packed_stride.hpp>

// #include "utils.h"

using namespace cute;

template <typename ElementOutput, typename TileShape, typename ClusterShape, typename MainloopScheduleType,
          bool WithBias>
void cutlass_int8_scaled_mm_sm90(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                                 const torch::Tensor& scales_a, const torch::Tensor& scales_b,
                                 const c10::optional<torch::Tensor>& bias) {
  using ArchTag = cutlass::arch::Sm90;

  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementCompute, ElementCompute,
                                                             Stride<Int<1>, Int<0>, Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementCompute, ElementCompute,
                                                             Stride<Int<0>, Int<1>, Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementOutput, ElementOutput,
                                                           Stride<Int<0>, Int<1>, Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  // Scale
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute,
                                                          cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  // With bias
  using ComputeWithBias = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiply_add, ElementOutput, ElementCompute,
                                                                 cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

  using EpilogueEVT = typename cutlass::platform::conditional<WithBias, EVTComputeWithBias, EVTCompute1>::type;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, ElementOutput, cutlass::layout::RowMajor, AlignmentC, ElementOutput,
      cutlass::layout::RowMajor, AlignmentOutput, EpilogueScheduleType, EpilogueEVT>::CollectiveOp;

  using Stages = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementInputA, cutlass::layout::RowMajor, AlignmentA, ElementInputB,
      cutlass::layout::ColumnMajor, AlignmentB, ElementAccumulator, TileShape, ClusterShape, Stages,
      MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                                          CollectiveMainloop, CollectiveEpilogue, TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = static_cast<ElementInputA*>(mat_a.data_ptr());
  auto b_ptr = static_cast<ElementInputB*>(mat_b.data_ptr());
  auto o_ptr = static_cast<ElementOutput*>(out.data_ptr());

  auto a_s_ptr = static_cast<ElementCompute*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementCompute*>(scales_b.data_ptr());

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename Gemm::Arguments args = {cutlass::gemm::GemmUniversalMode::kGemm,
                                   {m, n, k, 1},
                                   {a_ptr, stride_a, b_ptr, stride_b},
                                   {{},  // epilogue.thread
                                    nullptr,
                                    stride_c,
                                    o_ptr,
                                    stride_d}};

  if constexpr (WithBias) {
    ElementOutput* bias_ptr = static_cast<ElementOutput*>(bias->data_ptr());
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {bias_ptr},
        {},
    };
  } else {
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {},
    };
  }

  auto workspace = torch::empty(gemm_op.get_workspace_size(args),
                                torch::TensorOptions().dtype(torch::kUInt8).device(mat_a.device()));

  auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess,
              "gemm cannot implement, error: ", cutlassGetStatusString(can_implement));

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "gemm executioin failed, error: ", cutlassGetStatusString(status));
}

template <typename ElementOutput, typename TileShape, typename ClusterShape, typename MainloopScheduleType>
void sm90_int8_dispatch_bias(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                             const torch::Tensor& scales_a, const torch::Tensor& scales_b,
                             const c10::optional<torch::Tensor>& bias) {
  if (bias) {
    cutlass_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, true>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, false>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <typename ElementOutput>
void sm90_int8_dispatch_shape(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                              const torch::Tensor& scales_a, const torch::Tensor& scales_b,
                              const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 32) {
    if (n < 8192) {
      return sm90_int8_dispatch_bias<ElementOutput, Shape<_64, _64, _128>, Shape<_1, _8, _1>,
                                     cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b,
                                                                              bias);
    } else {
      return sm90_int8_dispatch_bias<ElementOutput, Shape<_64, _128, _128>, Shape<_1, _8, _1>,
                                     cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b,
                                                                              bias);
    }
  } else if (m <= 64) {
    if (n < 8192) {
      return sm90_int8_dispatch_bias<ElementOutput, Shape<_64, _64, _128>, Shape<_1, _4, _1>,
                                     cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b,
                                                                              bias);
    } else {
      return sm90_int8_dispatch_bias<ElementOutput, Shape<_64, _64, _256>, Shape<_1, _1, _1>,
                                     cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b,
                                                                              bias);
    }
  } else if (m <= 128) {
    if (n <= 4096) {
      return sm90_int8_dispatch_bias<ElementOutput, Shape<_64, _64, _128>, Shape<_2, _1, _1>,
                                     cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b,
                                                                              bias);
    } else {
      return sm90_int8_dispatch_bias<ElementOutput, Shape<_64, _128, _128>, Shape<_2, _1, _1>,
                                     cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b,
                                                                              bias);
    }
  } else {
    return sm90_int8_dispatch_bias<ElementOutput, Shape<_128, _128, _128>, Shape<_2, _1, _1>,
                                   cutlass::gemm::KernelTmaWarpSpecializedPingpong>(out, mat_a, mat_b, scales_a,
                                                                                    scales_b, bias);
  }
}

void sm90_int8_scaled_mm(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                         const torch::Tensor& scales_a, const torch::Tensor& scales_b, const torch::Dtype& out_dtype,
                         const c10::optional<torch::Tensor>& bias) {
  if (out_dtype == torch::kBFloat16) {
    sm90_int8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    sm90_int8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}
#endif
