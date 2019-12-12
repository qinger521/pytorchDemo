import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

#设定超参数（hyper parameters）
C = 3 # context window(周围多少个单词算周围)
K = 100 # number of negative samples
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 30000 # 词汇表大小
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100

def word_tokenize(text):
    return text.split()

# 创建一个单词表，将训练数据读入,添加UNK表示不常见单词
#需要建立从word到index的mapping，以及从index到mapping的映射
with open("text8/text8.train.txt","r") as fin:
    text = fin.read()
text = text.split()
# 选取文件中最常出现的单词
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab["<UNK>"] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = { word:i for i,word in enumerate(idx_to_word) }
word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_counts / np.sum(word_counts)
VOCAB_SIZE = len(idx_to_word)

#建立数据集
class WordEmbeaddingDataset(tud.Dataset):
    def __init__(self,text,word_to_idx,idx_to_word,word_freqs,word_counts):
        super(WordEmbeaddingDataset,self).__init__()
        self.text_encoded = [word_to_idx.get(word,word_to_idx["<UNK>"]) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C,idx)) + list(range(idx+1,idx+C+1)) # 周围词
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_word = torch.multinomial(self.word_freqs,K*pos_words.shape[0],True)
        return center_word,pos_words,neg_word

dataset = WordEmbeaddingDataset(text,word_to_idx,idx_to_word,word_freqs,word_counts)
dataloader = tud.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)

class EmbeddingModel(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size,self.embed_size)

    def forward(self, input_labels,pos_labels,neg_labels):
        #input_label:(batch_size)
        #pos_labels:(batch_size,(window_size * 2))
        #neg_labels:(batch_size,(window_size * 2 * k))
        input_embedding = self.in_embed(input_labels)  #batch_size * embed_size
        pos_embedding = self.in_embed(pos_labels)
        neg_embedding = self.in_embed(neg_labels)
        input_embedding = input_embedding.unsqueeze(2)
        pos_dot = torch.bmm(pos_embedding,input_embedding).squeeze(2)
        neg_dot = torch.bmm(neg_embedding,-input_embedding).squeeze(2)
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedings(self):
        return self.in_embed.weight.data.numpy()

model = EmbeddingModel(VOCAB_SIZE,EMBEDDING_SIZE)
optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i,(input_labels,pos_labels,neg_labels) in enumerate (dataloader):
        # print(input_labels,pos_labels,neg_labels)
        # if i>5:
        #     break
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        optimizer.zero_grad()
        loss = model(input_labels,pos_labels,neg_labels).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("epoch",e,"iteration",i,loss.item())
