import torch
from efficientformer_v2 import (
    EfficientFormer_depth,
    EfficientFormer_width,
    EfficientFormerV2,
    expansion_ratios_S2,
)
from torch import nn

NEW_ATTRIBUTES = [
    "Attractive",
    "Blurry",
    "Chubby",
    "Heavy_Makeup",
    "Male",
    "Oval_Face",
    "Pale_Skin",
    "Smiling",
    "Young",
    "Bald",
    "Bangs",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Receding_Hairline",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Hat",
    "Arched_Eyebrows",
    "Bags_Under_Eyes",
    "Bushy_Eyebrows",
    "Eyeglasses",
    "Narrow_Eyes",
    "Big_Nose",
    "Pointy_Nose",
    "5_o_Clock_Shadow",
    "Big_Lips",
    "Double_Chin",
    "Goatee",
    "Mouth_Slightly_Open",
    "Mustache",
    "No_Beard",
    "Sideburns",
    "Wearing_Lipstick",
    "High_Cheekbones",
    "Rosy_Cheeks",
    "Wearing_Earrings",
    "Wearing_Necklace",
    "Wearing_Necktie",
]


class MyModel(nn.Module):
    def __init__(self, batch_size=32) -> None:
        super().__init__()

        self.batch_size = batch_size

        efficientformer = EfficientFormerV2(
            layers=EfficientFormer_depth["S2"],
            embed_dims=EfficientFormer_width["S2"],
            downsamples=[True, True, True, True, True],
            vit_num=2,
            drop_path_rate=0.0,
            e_ratios=expansion_ratios_S2,
            num_classes=0,
            resolution=178,
            distillation=False,
        )
        self.network = nn.Sequential(efficientformer, Head(), nn.Sigmoid())

    # Regardless of model size results are the same: Vanishing gradient?

    def forward(self, x):
        return self.network(x)


def head_block(output_features):
    num_features = EfficientFormer_width["S2"][-1]
    return nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, output_features),
    )


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        groups = [9, 10, 5, 2, 9, 3, 2]
        self.blocks = nn.ModuleList([head_block(group) for group in groups])

    def forward(self, x):
        output = [block(x) for block in self.blocks]
        output = torch.cat(output, dim=1)
        return output
