import torch
import torch.backends.cudnn as cudnn

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
