import itertools

import pytest
import torch
from sgl_kernel import moe_sum_reduce


def moe_sum_reduce_torch(
    x: torch.Tensor, out: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)
    return out


@pytest.mark.parametrize(
    "num_tokens, topk, dim, scaling_factor, dtype",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024, 2048, 4096, 8192],  # num_tokens
            [1, 2, 4, 8, 16],  # topk
            [512, 1024, 2048, 4096, 8192],  # dim
            [0.1, 0.3, 0.8, 1.0],  # scaling_factor
            [torch.float16, torch.bfloat16],  # dtype
        )
    ),
)
def test_moe_reduce_sum(num_tokens, topk, dim, scaling_factor, dtype):

    input_tensor = torch.randn((num_tokens, topk, dim), dtype=dtype, device="cuda")
    out_sgl_kernel = torch.empty((num_tokens, dim), dtype=dtype, device="cuda")

    moe_sum_reduce(
        input_tensor,
        out_sgl_kernel,
        scaling_factor,
    )

    out_torch = torch.empty((num_tokens, dim), dtype=dtype, device="cuda")

    moe_sum_reduce_torch(
        input_tensor,
        out_torch,
        scaling_factor,
    )

    assert torch.allclose(
        out_torch, out_sgl_kernel, atol=1e-5, rtol=1e-2
    ), f"output mismatch: torch={out_torch}, SGLang={out_sgl_kernel}"


if __name__ == "__main__":
    pytest.main([__file__])
