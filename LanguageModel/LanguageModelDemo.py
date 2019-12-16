import torchtext
import torch.nn as nn
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

# w为了保证实验结果可以复现，经常将各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)

BATCH_SIZE = 32
# EMBEDDING_SIZE = 650 由于是在本机训练，所以可将EMBEDDING_SIZE设置的小一些
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
MAX_VOCAB_SIZE = 50000

# 使用text8作为训练数据，pytorch一个重要的概念是Field，它决定了你的数据如何处理
TEXT = torchtext.data.Field(lower=True)
# 构建数据集
train,val,test = torchtext.datasets.LanguageModelingDataset.splits(path="../text8"
                                                  ,train="text8.train.txt",validation="text8.dev.txt"
                                                  ,test="text8.test.txt",text_field=TEXT)
TEXT.build_vocab(train,max_size=MAX_VOCAB_SIZE)
# print(TEXT.vocab.itos[:10])
train_iter,val_iter,test_iter = torchtext.data.BPTTIterator.splits((train,val,test),batch_size=BATCH_SIZE
                                                                   ,device="cpu",bptt_len=50,repeat=False,shuffle=True)
# it = iter(train_iter)
# batch = next(it)
# print(batch)
class RNNModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super(RNNModel,self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size)
        self.linear = nn.Linear(hidden_size,vocab_size)

    def forward(self,text,hidden):
        emb = self.embed(text)
        output,hidden = self.lstm(emb,hidden)
        out_vocab = self.linear(output.view(-1,output.shape[2]))
        out_vocab = out_vocab.view(output.size(0),output.size(1),out_vocab.size(-1))
        return out_vocab,hidden

    def init_hidden(self,bsz,requires_grad=True):
        weight = next(self.parameters())
        return weight.new_zeros((1,bsz,self.hidden_size),requires_grad=True),weight.new_zeros((1, bsz, self.hidden_size), requires_grad=True)

model = RNNModel(vocab_size=len(TEXT.vocab),
                 embed_size=EMBEDDING_SIZE,
                 hidden_size=HIDDEN_SIZE
                 )

def repackage_hidden(h):
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h )

loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

VOCAB_SIZE = len(TEXT.vocab)

NUM_EPOCH = 2
GRAD_CLIP = 5.0
for epoch in range(NUM_EPOCH):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i , batch in enumerate(it):
        data,target = batch.text,batch.target
        hidden = repackage_hidden(hidden)
        output,hidden = model(data,hidden)

        loss = loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 100 == 0:
            print("loss",loss.item())

        if i % 10000:
            torch.save(model.state_dict(),"lm.pth")
