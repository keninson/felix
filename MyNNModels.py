import torch
from torch import nn


def eval_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


class LinearFunctionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


class QuadraticFunctionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight1 = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.weight2 = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight1 * (x**2) + self.weight2 * x + self.bias


class SinFunctionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.sin(self.b * x + self.c) + self.bias


# class SinFunctionModel(nn.Module()):
#     def __init__(self) -> None:
#         super().__init__()
#         self.layer1 = nn.Linear(in_features=1, out_features=3)
#         self.layer2 = nn.Linear(in_features=3, out_features=1)

#     def forward(self, x):
#         return self.layer1(x)


class CircleClassification0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
