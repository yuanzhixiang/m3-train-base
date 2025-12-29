# 导入 platform 模块，用于获取系统信息
import platform
# 导入 PyTorch 深度学习框架
import torch

# # 打印 CPU 架构
# output| python platform.machine(): arm64
print("python platform.machine():", platform.machine())

# 打印 PyTorch 版本号
# output| torch version: 2.9.1
print("torch version:", torch.__version__)

# 检查当前 PyTorch 是否编译时包含了 MPS 支持
# output| mps built: True
print("mps built:", torch.backends.mps.is_built())

# 检查 MPS 设备是否可用（需要硬件和软件都支持）
# output| mps available: True
print("mps available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    # 如果 MPS 可用
    # 在 MPS 设备上创建一个值为 1 的张量
    x = torch.ones(1, device="mps")

    # 打印张量内容和所在设备
    # output| tensor: tensor([1.], device='mps:0') device: mps:0
    print("tensor:", x, "device:", x.device)
else:
    # MPS 不可用时，打印提示信息
    print("MPS device not found. (Check macOS >= 12.3, arm64 env, torch install.)")
