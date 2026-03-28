import torch

# A single neuron
# y = w * x + b

# Input (2 features)
x = torch.tensor([1.0, 2.0])

# Weights (same size as input)
w = torch.tensor([0.5, -0.5])

# Bias
b = torch.tensor(0.0)

# Neuron output
y = torch.dot(w, x) + b

print(y)

# Adding activation
y = torch.relu(y)