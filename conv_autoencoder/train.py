from dataloader import *
from model import *
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os, time, datetime, json, pickle
import torch


def train_model(net, model_path, dataloaders_dict, criterion, optimizer, num_epochs, device, save_every):

    best_loss = 10e6
    records = []

    for epoch in range(num_epochs):
        net.to(device)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            
            dataloader = dataloaders_dict[phase]
            for item in dataloader:
                states, distance_m, map_m = item
                states = states.to(device).float()
                distance_m = distance_m.to(device).float().unsqueeze(1)
                map_m = map_m.to(device).float().unsqueeze(1)
                #print(states.shape, distance_m.shape, map_m.shape)
                x = states * distance_m * map_m

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    reconstruct_x = net(x)
                    loss = criterion(reconstruct_x, x)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() / np.prod(x.shape)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f}')
            records.append(epoch_loss)
            
        if epoch_loss < best_loss:
            for i in range(2):
                checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'records': np.array(records)
                }
                torch.save(checkpoint, os.path.join(model_path,'best_for_analysis_{}.pt'.format(i)))

                traced = torch.jit.trace(net.cpu(), torch.rand(1, 20, 32, 32))
                traced.save(os.path.join(model_path,'best_{}.pth'.format(i)))
            best_loss = epoch_loss

        '''
        if epoch % save_every == 0 or epoch + 1 == num_epochs:
            checkpoint = {
	        'epoch': epoch,
	        'state_dict': model.state_dict(),
	        'optimizer_state_dict': optimizer.state_dict(),
            'records': np.array(records)
            }
            torch.save(checkpoint, os.path.join(model_path,'{}.pt'.format(epoch)))
        '''
    return records

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using ', device)

    learning_rate = config['lr']
    load_prop = config['load_prop']
    batch_size = config['batch_size']
    save_every = config['save_every']
    num_epochs = config['epoch']

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_path = 'model_checkpoints/{}'.format(st)
    os.makedirs(model_path)

    episode_dir = '../data/episodes'
    obses, samples = create_dataset_from_json(episode_dir, load_prop=load_prop)

    train, val = train_test_split(samples, test_size=0.1, random_state=42)
    train_loader = DataLoader(
        LuxDataset(obses, train), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        LuxDataset(obses, val), 
        batch_size=batch_size, 
        shuffle=False
    )

    cae = Autoencoder().float().to(device)
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(cae.parameters(), lr=learning_rate)

    records = train_model(cae, model_path ,dataloaders_dict, criterion,optimizer,
        num_epochs=num_epochs, device=device, save_every=save_every)
    
    np.savetxt(os.path.join(model_path,'records.csv'), np.array(records), delimiter=",")
    with open(os.path.join(model_path,'config.json'), 'w') as f:
        json.dump(config, f)
    print('Work Done.')
