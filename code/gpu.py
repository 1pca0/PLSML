import torch
if torch.cuda.is_available():
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"发现 {num_gpus} 块可用的GPU.")
else:
    print("未发现可用的GPU.")