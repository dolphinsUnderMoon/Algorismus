import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        # input shape: (227, 227, 3)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=(11, 11),
                stride=4
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)  # (27, 27, 96)
            # LRN
        )

        # (27, 27, 96)
        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=(5, 5),
                padding=2,
                groups=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)  # (13, 13, 256)
            # LRN
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.ReLU(inplace=True)  # (13, 13, 384)
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.ReLU(inplace=True)  # (13, 13, 384)
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)  # (6, 6, 256)
        )

        self.layer_6 = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()  # (4096)
        )

        self.layer_7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()  # (4096)
        )

        self.layer_8 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.layer_5(
            self.layer_4(
                self.layer_3(
                    self.layer_2(
                        self.layer_1(x)))))
        x = x.view(x.size(0), -1)
        x = self.layer_6(x)
        x = self.layer_7(x)
        output = self.layer_8(x)

        return output


if __name__ == '__main__':
    alexnet = AlexNet(num_classes=10)
    print(alexnet)

    random_input = Variable(torch.randn(1, 3, 227, 227))
    output = alexnet(random_input)
    print(output)