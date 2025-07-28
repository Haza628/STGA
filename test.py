import torch
import torch.nn as nn
import torch.optim as optim

# 假设 self._xyz 是您的数据张量
_xyz = torch.randn(10, requires_grad=True).cuda()
gaussian_indices = torch.tensor([True, False, True, False, True, False, True, False, True, False]).cuda()

# 将需要优化的部分克隆并移动到 GPU 上
_xyz_training = _xyz[gaussian_indices].clone().cuda()

# 定义优化器，这里使用 SGD 作为示例
optimizer = optim.SGD([_xyz_training], lr=0.001)

num_epochs = 100
# 模拟训练循环
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 计算损失，这里假设损失函数为 self._xyz_training 的平方和
    loss = torch.sum(_xyz_training ** 2)
    
    # 手动计算梯度
    loss.backward()
    
    # 在优化器中使用手动计算的梯度
    optimizer.step()

    _xyz[gaussian_indices] = _xyz_training
    print(_xyz)
    print(_xyz_training)