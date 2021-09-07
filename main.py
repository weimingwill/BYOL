import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets
import numpy as np
from collections import defaultdict

from modules import BYOL
from modules.transformations import TransformsSimCLR

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def cleanup():
    dist.destroy_process_group()


def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(0)

    # dataset
    train_dataset = datasets.CIFAR10(
        args.dataset_dir,
        download=True,
        transform=TransformsSimCLR(size=args.image_size), # paper 224
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # model
    if args.resnet_version == "resnet18":
        resnet = models.resnet18(pretrained=False)
    elif args.resnet_version == "resnet50":
        resnet = models.resnet50(pretrained=False)
    else:
        raise NotImplementedError("ResNet not implemented")

    model = BYOL(resnet, image_size=args.image_size, hidden_layer="avgpool")
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # solver
    global_step = 0
    for epoch in range(args.num_epochs):
        metrics = defaultdict(list)
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            loss = model(x_i, x_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()  # update moving average of target encoder

            if step % 1 == 0:
                print(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")

            metrics["Loss/train"].append(loss.item())
            global_step += 1

        # write metrics to TensorBoard
        print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

        if epoch % args.checkpoint_epochs == 0:
            print(f"Saving model at epoch {epoch}")
            torch.save(resnet.state_dict(), f"./model-{epoch}.pt")


    # save your improved network
    torch.save(resnet.state_dict(), "./model-final.pt")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=224, type=int, help="Image size")
    parser.add_argument(
        "--learning_rate", default=3e-4, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=192, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resnet_version", default="resnet18", type=str, help="ResNet version."
    )
    parser.add_argument(
        "--checkpoint_epochs",
        default=5,
        type=int,
        help="Number of epochs between checkpoints/summaries.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./datasets",
        type=str,
        help="Directory where dataset is stored.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers (caution with nodes!)",
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes",
    )
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")
    args = parser.parse_args()

    # Master address for distributed data parallel
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = "8010"
    # args.world_size = args.gpus * args.nodes

    # Initialize the process and join up with the other processes.
    # This is “blocking,” meaning that no process will continue until all processes have joined.
    main(args)
    # mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
