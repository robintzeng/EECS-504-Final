from tqdm import tqdm


def train(model, optimizer, loader, epoch, device):
    pbar = tqdm(iter(loader))
    model.train()
    
    for rgb, lidar, mask, gt, params in pbar:
        rgb, lidar, mask = rgb.to(device), lidar.to(device), mask.to(device)
        gt, params = gt.to(device), params.to(device) 
        print(rgb.size(), lidar.size(), gt.size())

        model(rgb, lidar, mask)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()