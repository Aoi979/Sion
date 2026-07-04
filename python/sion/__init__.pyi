from typing import Any

import torch

_C: Any
ops: Any

def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = ...,
    beta: float = ...,
) -> torch.Tensor: ...

def sgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = ...,
    beta: float = ...,
) -> torch.Tensor: ...

def hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = ...,
    beta: float = ...,
) -> torch.Tensor: ...

def hgemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = ...,
    beta: float = ...,
) -> torch.Tensor: ...

def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor: ...
