import torch

if torch.cuda.is_available():
    Dvc = torch.device("cuda:0")
    print("Running on GPU cuda:0")
else:
    Dvc = torch.device("cpu")
    print("Running on CPU")

CpuDvc = torch.device("cpu")


torch_pi = torch.acos(torch.zeros(1)).item() * 2
