import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, input_channels, output_channels,
                 kernel_size=None, stride=None,num_routing_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.num_routing_iterations = num_routing_iterations

        if self.num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(self.num_capsules,
                                                          self.num_route_nodes,
                                                          input_channels,
                                                          output_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels=input_channels,
                           out_channels=output_channels,
                           kernel_size=kernel_size,
                           stride=stride, padding=0) for _ in range(self.num_capsules)]
            )

    @staticmethod
    def squash(tensor, dim=-1):
        square_l2_norm_value = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = square_l2_norm_value / (1 + square_l2_norm_value)
        return scale * (tensor / torch.sqrt(square_l2_norm_value))

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            temp_route_parameter = Variable(torch.zeros(*priors.size()))

            if torch.cuda.is_available():
                temp_route_parameter = temp_route_parameter.cuda()

            for i in range(self.num_routing_iterations):
                routing_prob_distribution = F.softmax(temp_route_parameter, dim=2)
                _outputs = self.squash((routing_prob_distribution * priors).sum(dim=2, keepdim=True))

                if i != self.num_routing_iterations - 1:
                    temp_route_parameter += (priors * _outputs).sum(dim=-1, keepdim=True)
        else:
            _outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            _outputs = torch.cat(_outputs, dim=-1)
            _outputs = self.squash(_outputs)

        return _outputs


if __name__ == "__main__":
    primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, input_channels=256, output_channels=32,
                                    kernel_size=9, stride=2)
    print(primary_capsules)
    digit_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, input_channels=8,
                                  output_channels=16)
    print(digit_capsules)
