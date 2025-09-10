import torch
from torch import nn


class MyMLP(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(2 * 3, 10), 
            nn.ReLU(), 
            nn.Linear(10, 4)
        )

        # `__init__()` done
        pass


    def forward(
        self, 

        model_input_list: torch.FloatTensor
    ) -> torch.FloatTensor:
        model_output_list = self.flatten(model_input_list)

        model_output_list = self.mlp(model_output_list)

        # `forward()` done
        return model_output_list


if __name__ == "__main__":
    my_mlp = MyMLP()

    batch_size = 5
    model_input_list = torch.randn(
        (batch_size, 2, 3)
    )

    model_output_list = my_mlp(model_input_list)

    print(model_output_list.shape)
