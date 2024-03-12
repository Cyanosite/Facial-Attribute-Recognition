from torch import nn
import torch
from efficientformer_v2 import EfficientFormerV2, EfficientFormer_depth, EfficientFormer_width, expansion_ratios_S2

class EfficientFormerV3(nn.Module):
    def __init__(self, batch_size = 32) -> None:
        super().__init__()

        self.batch_size = batch_size

        efficientformer = EfficientFormerV2(
            layers = EfficientFormer_depth['S2'],
            embed_dims=EfficientFormer_width['S2'],
            downsamples=[True, True, True, True, True],
            vit_num=2,
            drop_path_rate=0.0,
            e_ratios=expansion_ratios_S2,
            num_classes=0,
            resolution=178,
            distillation=False
        )
        self.network = nn.Sequential(
            efficientformer,
            Head()
        )


    def forward(self, x):
        return self.network(x)

def head_block(output_features):
    num_features = EfficientFormer_width['S2'][-1]
    return nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
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

def train_loop(dataloader, model, loss_fn, optimizer, device, epochs = 5):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(1, epochs+1):
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
                print(f"Epoch: {epoch}/{epochs} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    #size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    correct_attributes = [0] * 40
    baseline = [0] * 40

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = transform_y(y)
            y = y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            pred_binary = (pred > 0.5).type(torch.float)
            correct += ((pred_binary == y) & (y == 0) | (pred_binary == y) & (y == 1)).type(torch.float).sum().item()
            correct_attributes += (pred_binary == y).type(torch.float).sum(dim=0).cpu().numpy()
            baseline += y.sum(dim=0).cpu().numpy()

    test_loss /= num_batches
    correct /= num_batches * model.batch_size * 40
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (correct_attributes / (num_batches * model.batch_size), baseline / (num_batches * model.batch_size))


def transform_y(y):
    mapping = [26, 19, 0, 20, 9, 10, 27, 24, 11, 12, 1, 13, 21, 2, 28, 22, 29, 14, 3, 35, 4, 30, 31, 23, 32, 5, 6, 25, 15, 36, 33, 7, 16, 17, 37, 18, 34, 38, 39, 8]
    for batch, x in enumerate(y):
        new_batch = torch.zeros(40, dtype=torch.float32)
        for i, to in enumerate(mapping):
            new_batch[to] = x[i]
        y[batch] = new_batch
    return y.to(torch.float32)
