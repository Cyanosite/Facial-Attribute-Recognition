from torch import nn
import torch
from efficientformer_v2 import EfficientFormerV2, EfficientFormer_depth, EfficientFormer_width, expansion_ratios_S0

class EfficientFormerV3(nn.Module):
    def __init__(self, batch_size = 32) -> None:
        super().__init__()

        self.batch_size = batch_size

        efficientformer = EfficientFormerV2(
            layers = EfficientFormer_depth['S0'],
            embed_dims=EfficientFormer_width['S0'],
            downsamples=[True, True, True, True, True],
            vit_num=2,
            drop_path_rate=0.0,
            e_ratios=expansion_ratios_S0,
            num_classes=0,
            resolution=218,
            distillation=False
        )
        self.network = nn.Sequential(
            efficientformer,
            Head()
        )


    def forward(self, x):
        return self.network(x)

def head_block(output_features):
    num_features = EfficientFormer_width['S0'][-1]
    return nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, output_features),
        nn.Sigmoid()
    )


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        groups = [9, 10, 5, 2, 9, 3, 2]
        self.blocks = nn.ModuleList(
            [head_block(group) for group in groups]
        )

    def forward(self, x):
        output = [block(x) for block in self.blocks]
        output = torch.cat(output, dim=1)
        return output

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = transform_y(y)
        y = y.to(device)
    
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * model.batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = transform_y(y)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def transform_y(y):
    mapping = {2: 0, 10: 1, 13: 2, 18: 3, 20: 4, 25: 5, 26: 6, 31: 7, 39: 8, 4: 9, 5: 10, 8: 11, 9: 12, 11: 13, 17: 14, 28: 15, 32: 16, 33: 17, 35: 18, 1: 19, 3: 20, 12: 21, 15: 22, 23: 23, 7: 24, 27: 25, 0: 26, 6: 27, 14: 28, 16: 29, 21: 30, 22: 31, 24: 32, 30: 33, 36: 34, 19: 35, 29: 36, 34: 37, 37: 38, 38: 39}
    #attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    #new_attributes = ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 'Oval_Face', 'Pale_Skin', 'Smiling', 'Young', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses', 'Narrow_Eyes', 'Big_Nose', 'Pointy_Nose', '5_o_Clock_Shadow', 'Big_Lips', 'Double_Chin', 'Goatee', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Sideburns', 'Wearing_Lipstick', 'High_Cheekbones', 'Rosy_Cheeks', 'Wearing_Earrings', 'Wearing_Necklace', 'Wearing_Necktie']
    #y = {attributes.index(v): new_attributes.index(v) for k, v in y.items()}
    for batch, x in enumerate(y):
        new_batch = torch.zeros(40, dtype=torch.float32)
        for i in range(40):
            new_batch[mapping[i]] = x[i]
        y[batch] = new_batch
    return y.to(torch.float32)
