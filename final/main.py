import torch
from dataloader.dataloader import get_loader
import argparse
from tqdm import tqdm
from training.train import train
from model.DeepLidar import deepLidar, test_model
import torch.optim as optim

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




def main():
    train_loader = get_loader('train')
    model = deepLidar().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    for epoch in range(args.epoch):
        train(model, optimizer, train_loader, epoch, DEVICE)


if __name__ == '__main__':
    main()