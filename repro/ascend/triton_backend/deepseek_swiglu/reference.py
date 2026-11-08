import torch
import torch_npu

def swi_glu(input: torch.Tensor, dim: int = -1):
    return torch_npu.npu_swiglu(input, dim=dim)
