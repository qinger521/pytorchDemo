import torch
'''
    pytorch是一个给予python的科学计算包
'''

# 定义一个空矩阵
x = torch.empty(5,3)
print(x)

# 构造一个随即初始化矩阵
x = torch.rand(5,3)
print(x)

# 构造一个矩阵全为0，且数据类型为long
x = torch.zeros(5,3,dtype=torch.long)
print(x)

# 构造一个张量，直接使用数据
x = torch.tensor([5,5,3])
print(x)

# 获取张量的维度信息
print(x.size())

y = torch.rand(5,3)
x = torch.rand(5,3)

# 张量的加法：提供一个tensor作为参数
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

# 加法：in-place
y.add_(x)
print(y)

# pytorch支持标准的numpy的索引操作
print(x[:,1])