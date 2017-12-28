import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3))
        self.conv_2 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=(3, 3))
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        output = self.pool_3(x)

        return output


class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()

        self.block_1 = Block()
        self.block_2 = Block()
        self.block_2.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=(3, 3)
        )

        self.fc_1 = nn.Linear(75, 10)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = x.view(x.size(0), -1)
        output = self.fc_1(x)

        return output


if __name__ == '__main__':
    simplevgg = SimpleVGG()
    print(simplevgg)

    random_input = Variable(torch.randn(1, 1, 32, 32))
    output = simplevgg(random_input)
    print(output)