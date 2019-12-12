#使用pytorch实现fizz——buzz游戏，3的倍数为fizz，5的倍数为buzz，15的倍数为fizzbuzz
import numpy as np
import torch

NUM_DIGITS = 10

def fizz_buzz_encode(i):
    if i%15 == 0: return 3
    elif i%5 == 0: return 2
    elif i%3 == 0: return 1
    else:return 0

#将数字转化为二进制，神经网络更好训练
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])

def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]

trX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(101,2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2 ** NUM_DIGITS)])

#使用pytorch定义模型
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS,NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN,4)
)
#定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()
#定义优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)
#模型训练
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0,len(trX),BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]
        y_pred = model(batchX)
        loss = loss_fn(y_pred,batchY)
        print("Epoch",epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
testX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(1,101)])
with torch.no_grad():
    testY = model(testX)
predicts = zip(range(1,101),testY.max(1)[1].cpu().data.tolist())
print([fizz_buzz_decode(i,x) for i,x in predicts])