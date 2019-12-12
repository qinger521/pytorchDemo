import torch
import torch.nn as nn

N,D_in,H,D_out = 64,1000,100,10
x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

#原生方法
#w1 = torch.randn(D_in,H,requires_grad=True)
#w2 = torch.randn(H,D_out,requires_grad=True)

#使用pytorch则可使用nn中的model,sequential表示将一串模型拼接起来
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),   # w_1 * x + b_1
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out)
)
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
for t in range(500):
    #1.原生方法：
    #forward pass
    #h = x.mm(w1) # N*H
    #h_relu = h.clamp(min=0) #激活函数
    #y_pred = h_relu.mm(w2)

    #2.使用pytorch
    y_pred = model(x) #model.forward()

    #原生方法：
    #compute loss
    ##loss = (y_pred - y).pow(2).sum()
    #print(t,loss.item())

    loss = loss_fn(y_pred,y)
    print(t, loss.item())
    model.zero_grad()
    #backward pass
    #compute the gradient
    # 优化器方法：
    optimizer.zero_grad()
    loss.backward()

    # 优化器方法：
    optimizer.step()


    #update weights of w1 and w2
    ##with torch.no_grad():
    #    for param in model.parameters():
    #        param -= learning_rate * param.grad
