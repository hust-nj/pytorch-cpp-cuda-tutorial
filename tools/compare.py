from models.lltm import LLTM as LLTM_NV
from models.lltm_cpp import LLTM as LLTM_CPP
from models.lltm_cuda import LLTM as LLTM_CUDA

import time

import torch

cuda_device = torch.device("cuda")

batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features, device=cuda_device)
h = torch.randn(batch_size, state_size, device=cuda_device)
C = torch.randn(batch_size, state_size, device=cuda_device)


def test_speed(rnn, name):
    rnn.to(cuda_device)
    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        torch.cuda.synchronize()
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        torch.cuda.synchronize()
        backward += time.time() - start

    print(name,'Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))

test_speed(LLTM_NV(input_features, state_size), 'naive lltm')
test_speed(LLTM_CPP(input_features, state_size), 'cpp lltm')
test_speed(LLTM_CUDA(input_features, state_size), 'cuda_kernel lltm')
