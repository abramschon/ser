
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(
    dir : str, 
    train : bool,
    transform : transforms.Compose,
    batch_size : int,
    ):

    return DataLoader(
        datasets.MNIST(root=dir, download=True, train=train, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        )
