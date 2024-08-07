import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pennylane as qml
from torchsummary import summary


class PatchQuantumGenerator(nn.Module):
    def __init__(self, n_generators, device, n_qubits=5, n_a_qubits=1, q_depth=6, n_generators=4, q_delta=1):
        """ n_generators (int): Number of sub-generators
            n_qubits (int): Total number of qubits / N
            n_a_qubits (int): Number of ancillary qubits / N_A
            q_depth (int): Depth of the parameterised quantum circuit / D
            device (string): Device; gpu, if available, else cpu.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators
        self.device = device

    @qml.qnode(dev, diff_method="parameter-shift")
    def quantum_circuit(self, noise, weights):
        weights = weights.reshape(q_depth, n_qubits)
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)
        # For each layer
        for i in range(q_depth):
            # RY Gates
            for y in range(n_qubits):
                qml.RY(weights[i][y], wires=y)
            # Control Z gates
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
        return qml.probs(wires=list(range(n_qubits)))

    def partial_measure(self, noise, weights):
        probs = quantum_circuit(noise, weights)
        probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
        probsgiven0 /= torch.sum(probs)

        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven

    def forward(self, x):
        patch_size = 2 ** (n_qubits - n_a_qubits)
        images = torch.Tensor(x.size(0), 0).to(self.device)
        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(self.device)
            for elem in x:
                q_out = self.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            images = torch.cat((images, patches), 1)
        return images


class Discriminator(nn.Module):
    def __init__(self, input_shape=(1, 8, 8), n_classes=2):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8), n_classes)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class DiscriminatorSupervised(nn.Module):
    def __init__(self, discriminator):
        super(DiscriminatorSupervised, self).__init__()
        self.discriminator = discriminator
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.discriminator(x)
        x = self.softmax(x)
        return x


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        Z_x = torch.sum(torch.exp(x), dim=-1, keepdim=True)
        D_x = Z_x / (Z_x + 1)
        return D_x


class DiscriminatorUnsupervised(nn.Module):
    def __init__(self, discriminator):
        super(DiscriminatorUnsupervised, self).__init__()
        self.discriminator = discriminator
        self.custom_activation = CustomActivation()

    def forward(self, x):
        x = self.discriminator(x)
        x = self.custom_activation(x)
        return x


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discriminator_base = Discriminator()
    discriminator_unsup = DiscriminatorUnsupervised(discriminator_base).to(device)
    discriminator_sup = DiscriminatorSupervised(discriminator_base).to(device)
    generator = PatchQuantumGenerator(n_generators, device).to(device)
    print("Input shape: (1, 8, 8)")
    print("Base Discriminator:", summary(discriminator, (1, 8, 8)), sep='\n')
    print("Supervised Discriminator", summary(disc_sup, (1, 8, 8)), sep='\n')
    print("Unsupervised Discriminator", summary(disc_unsup, (1, 8, 8)), sep='\n')
    print("Generator", summary(generator, (1, 8, 8)), sep='\n')
