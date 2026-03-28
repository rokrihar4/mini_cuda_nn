A good way to do this is to not start by building “mini PyTorch.” Start by building a tiny autograd engine with just enough CUDA to feel the hard parts.

The project should look like this:

a Tensor object
a computation graph
a few ops with forward/backward
CUDA-backed storage
a tiny MLP trained on toy data

That is enough to learn a lot.

What you are actually building

Your framework needs only a few concepts:

1. Tensor
stores data pointer
shape
gradient
device (cpu or cuda)
link to how it was created
Operation
forward
backward
Autograd engine
tracks graph
topologically sorts nodes
calls backward in reverse order
CUDA backend
allocate GPU memory
launch kernels for ops
copy data CPU ↔ GPU
Module / Parameter
linear layer weights and bias
optimizer step

Do not start with convolutions, broadcasting, fancy slicing, or dynamic shapes.

Best scope for v1

Build only this:

tensor creation
add
mul
matmul
relu
sum
mean
backward
SGD
Linear layer
MSE loss

With that, you can train:
linear regression
XOR with a 2-layer MLP
tiny spiral classifier

That is plenty.

Recommended architecture

Use this layout:

mini_cuda_nn/
├── python/
│   ├── tensor.py
│   ├── autograd.py
│   ├── nn.py
│   ├── optim.py
│   └── train_xor.py
├── cpp/
│   ├── bindings.cpp
│   ├── tensor_cuda.cu
│   ├── ops_cuda.cu
│   └── ops.h
├── setup.py
└── README.md
Why this split works
Python handles user-facing API and graph logic
C++/CUDA handles fast kernels and memory ops

This mirrors how serious frameworks are layered.