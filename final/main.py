import torch
from dataloader.dataloader import get_loader
import argparse
from tqdm import tqdm
from training.train import train, val, EarlyStop
from model.DeepLidar import deepLidar
import torch.optim as optim
import os

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model', help='saved_model_name')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVED_MODEL_PATH = os.path.join('saved_model', args.saved_model_name)


def main():
    train_loader = get_loader('train')
    val_loader = get_loader('train')

    model = deepLidar().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    early_stop = EarlyStop(saved_model_path=SAVED_MODEL_PATH+'.tar', patience=10, mode='min')

    for epoch in range(args.epoch):
        train(model, optimizer, train_loader, epoch, DEVICE)
        loss = val(model, optimizer, val_loader, epoch, DEVICE)

        if early_stop.stop(loss, model, epoch+1):
            break

if __name__ == '__main__':
    main()