import torch
from model.lltm import LLTM

input_features = 16
state_size = 16
batch_size = 4

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

new_h, new_C = rnn(X, (h, C))

print(new_h.size())
print(new_C.size())
