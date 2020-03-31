import torch
from dataloader.dataloader import get_loader
import argparse
from tqdm import tqdm
from training.train import train_val, EarlyStop
from model.DeepLidar import deepLidar
import torch.optim as optim
import os

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model', help='saved_model_name')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-s', '--stage', type=str, default='A', help='decide stage to train network')
parser.add_argument('-l', '--load_model', help='load model')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'
STAGE = args.stage.upper()
SAVED_MODEL_PATH = os.path.join('saved_model', args.saved_model_name+STAGE)


def main():
    loader = {'train': get_loader('train'), 'val': get_loader('val', shuffle=False)}

    model = deepLidar().to(DEVICE)

    if args.load_model:
        dic = torch.load(args.loadmodel)
        state_dict = dic["state_dict"]
        print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))
        model.load_state_dict(model_dict)

    early_stop = EarlyStop(saved_model_path=SAVED_MODEL_PATH, patience=10, mode='min')

    for epoch in range(args.epoch):
        loss = train_val(model, loader, epoch, DEVICE, STAGE)
        if early_stop.stop(loss, model, epoch+1):
            break

if __name__ == '__main__':
    main()