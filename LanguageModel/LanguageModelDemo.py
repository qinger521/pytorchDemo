import torchtext
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