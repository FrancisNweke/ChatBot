import torch
import torch.nn as nn


class NeuralModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = output_size

        # Define network layers
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.num_classes)
        self.relu = nn.ReLU()

    # Define feed forward network
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out
        # No softmax activation because PyTorch's CrossEntropy applies it for us.
