import os
import sys
from collections import Counter
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
#英文分词
import nltk
# nltk.download('stopwords')
# 读入文件
def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file,'r') as f:
        for line in f :
            line = line.strip().split("\t")
            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
    return en , cn

# train_file = "../Data/PyTorch-Course-master/notebooks/nmt/en-cn/train.txt"
train_file = "train.txt"
dev_file = "dev.txt"
train_en,train_cn = load_data(train_file)
dev_en,dev_cn = load_data(dev_file)

# 构建单词表
UNK_IDX = 0
PAD_IDX = 1
def build_dict(sentences,max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    # 按照出现频率
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 2
    word_dict = { w[0]: index+2 for index,w in enumerate(ls)}
    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    return word_dict,total_words
en_dict,en_total_words = build_dict(train_en)
cn_dict,cn_total_words = build_dict(train_cn)
# 解码
inv_en_dict = {v:k for k,v in en_dict.items()}
inv_cn_dict = {v:k for k,v in cn_dict.items()}

# encode the sequence
def encode(en_sentences,cn_sentences,en_dict,cn_dict,sort_by_len=True):
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w,0) for w in sent] for sent in en_sentences ]
    out_cn_sentences = [[cn_dict.get(w,0) for w in sent] for sent in cn_sentences ]
    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)),key=lambda x:len(seq[x]))
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index ]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index ]
    return out_en_sentences,out_cn_sentences
train_en,train_cn = encode(train_en,train_cn,en_dict,cn_dict)
dev_en,dev_cn = encode(dev_en,dev_cn,en_dict,cn_dict)

# 将句子分为batch
def get_minibatches(n,minibatch_size,shuffle=True):
    idx_list = np.arange(0,n,minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx,min(idx+minibatch_size,n)))
    return minibatches

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)

    x = np.zeros((n_samples,max_len)).astype('int32')
    x_lengths = np.array(lengths).astype("int32")
    for idx,seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x,x_lengths

def gen_examples(en_sentences,cn_sentences,batch_size):
    minibatches = get_minibatches(len(en_sentences),batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_x,mb_x_len = prepare_data(mb_en_sentences)
        mb_y,mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x,mb_x_len,mb_y,mb_y_len))
    return all_ex

# 获得数据集
batch_size = 64
train_data = gen_examples(train_en,train_cn,batch_size)
random.shuffle(train_data)
dev_data = gen_examples(dev_en,dev_cn,batch_size)


# 至此数据处理完成，开始搭建seq2seq
# 无attention版本
class PlainEncoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,dropout=0.2):
        super(PlainEncoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,lengths):
        #将batch里面的seq按照长度排序
        sorted_len,sorted_idx = lengths.sort(0,descending=True)
        x_sorted = x[sorted_idx]
        embedded = self.dropout(self.embed(x_sorted))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,sorted_len.long().data.numpy(),batch_first=True)
        packed_out,hid = self.rnn(packed_embedded)
        out , _ = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True)

        _,original_idx = sorted_idx.sort(0,descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        return out,hid[[-1]]

class PlainDecoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,dropout=0.2):
        super(PlainDecoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.GRU(2*hidden_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(2*hidden_size,vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,y,y_lengths,hid):
        # 将batch里面的seq按照长度排序
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        #         print(output_seq.shape)
        hid = hid[:, original_idx.long()].contiguous()

        output = F.log_softmax(self.out(output_seq), -1)

        return output, hid
        # sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        # y_sorted = y[sorted_idx.long()]
        # hid = hid[:, sorted_idx.long()]
        # embedded = self.dropout(self.embed(y_sorted))
        #
        # embedded = torch.cat([embedded,hid.unsqueeze(0).expand_as(embedded)],2)
        #
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().data.numpy(), batch_first=True)
        # packed_out, newhid = self.rnn(packed_embedded,hid)
        # out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        #
        # _, original_idx = sorted_idx.sort(0, descending=False)
        # out = out[original_idx.long()].contiguous()
        # newhid = newhid[original_idx.long()].contiguous()
        #
        # out = torch.cat([out, hid.unsqueeze(0).expand_as(embedded)], 2)
        # out = F.log_softmax(self.fc(out))
        return out, newhid


class PlainSeq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(PlainSeq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,x_lengths,y,y_lengths):
        encoder_out,hid = self.encoder(x,x_lengths)
        output,hid = self.decoder(y,y_lengths,hid)
        return output,None


# 损失函数
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion,self).__init__()
    def forward(self,input,target,mask):
        input = input.contiguous().view(-1,input.size(2))
        target = target.contiguous().view(-1,1)
        mask = mask.contiguous().view(-1,1)
        output = -input.gather(1,target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout = 0.2
hidden_size = 100
encoder = PlainEncoder(vocab_size=en_total_words,hidden_size=hidden_size,dropout=dropout)
decoder = PlainDecoder(vocab_size=cn_total_words,hidden_size=hidden_size,dropout=dropout)
model = PlainSeq2Seq(encoder,decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())

# 训练代码
def train(model,data,num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = num_words = total_loss = 0.
        for it , (mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:,1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred,attn = model(mb_x,mb_x_len,mb_y,mb_y_len)

            mb_out_mask = torch.arrange(mb_y_len.max().item(),device=device)[None,:] < mb_y_len[:,None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred,mb_output,mb_out_mask)
            num_words += torch.sum(mb_y_len).item()
            total_loss += loss.item()*num_words
            total_num_words += num_words

            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameter(),5.)
            optimizer.step()

            if it % 100 == 0:
                print("Epoch",epoch,"iteration",it,"loss",loss.item())
        print("Epoch",epoch,"Training loss",total_loss/total_num_words)

train(model,train_data,num_epochs=2)