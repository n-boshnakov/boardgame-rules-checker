import torch

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('No GPU detected - LLM will run on CPU (very slow)')
