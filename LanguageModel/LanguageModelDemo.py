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
