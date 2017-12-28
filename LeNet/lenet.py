import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=6,
                                kernel_size=(5, 5))
        self.conv_2 = nn.Conv2d(in_channels=6,
                                out_channels=16,
                                kernel_size=(5, 5))
        self.fc_1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc_2 = nn.Linear(in_features=120, out_features=84)
        self.fc_3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv_1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv_2(x)), kernel_size=(2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        output = self.fc_3(x)

        return output


if __name__ == "__main__":
    lenet = LeNet()
    print(lenet)

    random_input = Variable(torch.randn(1, 1, 32, 32))
    output = lenet(random_input)
    print(output)