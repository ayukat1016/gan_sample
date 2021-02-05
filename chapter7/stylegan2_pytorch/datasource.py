import os
import torch
from dataset import TrainDataset
import torchvision.transforms as transforms
from transform import TranformDynamicRange

def get_dataloader(data_path, resolution, batch_size):
    dataset = TrainDataset(
        file_path=os.path.join(data_path, '*.png'),
        transform=transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.ToTensor(),
                TranformDynamicRange([0, 255], [-1, 1])
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader
