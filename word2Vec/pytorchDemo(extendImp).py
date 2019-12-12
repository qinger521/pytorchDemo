import torch
import torch.nn as nn

N,D_in,H,D_out = 64,1000,100,10
x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(D_in,H)
        self.linear2 = torch.nn.Linear(H,D_out)


    def forward(self,x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred

model = TwoLayerNet(D_in,H,D_out)
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
for t in range(500):
    y_pred = model(x) #model.forward()
    loss = loss_fn(y_pred,y)
    print(t, loss.item())
    model.zero_grad()
    #backward pass
    # 优化器方法：
    optimizer.zero_grad()
    loss.backward()
    # 优化器方法：
    optimizer.step()