import torch

tensor = torch.randn(1000, 1000)  

if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print("Tensor is on device:", tensor.device)
else:
    print("CUDA is not available. Tensor is on:", tensor.device)
