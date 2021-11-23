from pathlib import Path
import torch
from torch import optim

from ser.model import Net
from ser.transforms import get_transforms
from ser.data import get_data_loader
from ser.train import train_model

import typer
import json

main = typer.Typer() #

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    # Typer options - make command like arguments better
    # `...` makes this option mandatory (i.e. users have to pass in this argument)
    # `help` info that gets printed out if you run `python cli.py --help` 
    name: str = typer.Option( 
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option( 
        10, "-e", "--epochs", help="Number of epochs to train model."
    ),
    batch_size: int = typer.Option( 
        32, "-b", "--batch_size", help="Size of each batch."
    ),
    learning_rate: float = typer.Option( 
        1e-3, "-l", "--learning_rate", help="Model's learning rate."
    )
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # save the parameters!
    with open(f"{DATA_DIR}/{name}.json" , "w") as f:
        f.write(
            json.dumps((epochs,batch_size,learning_rate))
        )

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = get_transforms()

    # dataloaders
    training_dataloader = get_data_loader(DATA_DIR, True, ts, batch_size)
    validation_dataloader = get_data_loader(DATA_DIR, False, ts, batch_size)

    # train model
    train_model(model, optimizer, 
                training_dataloader, validation_dataloader,
                epochs, device)


@main.command()
def infer():
    print("This is where the inference code will go")
