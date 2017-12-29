import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        # (224, 224, 3)
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # (112, 112, 64)
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # (56, 56, 128)
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # (28, 28, 256)
        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # (14, 14, 512)
        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # (7, 7, 512)
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.Dropout()
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout()
        )

        self.fc_3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1000),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.block_5(
            self.block_4(
                self.block_3(
                    self.block_2(
                        self.block_1(x)
                    )
                )
            )
        )

        x = x.view(x.size(0), -1)

        output = self.fc_3(self.fc_2(self.fc_1(x)))

        return output


if __name__ == '__main__':
    vgg19 = VGG19()
    print(vgg19)

    random_input = Variable(torch.randn(1, 3, 224, 224))
    output = vgg19(random_input)
    print(output)