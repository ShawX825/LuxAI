from dataloader import *
from model import *
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os, time, datetime, json, pickle

SEED = 42

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_model(net, model_path, dataloaders_dict, criterion,
 optimizer, num_epochs, device, save_every):

    best_acc = 0.0
    records = []

    for epoch in range(num_epochs):
        net.to(device)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for item in dataloader:
                states, actions = item
                states = states.to(device).float()
                #states = states * distance_m * map_m
                actions = actions.to(device).long()               
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    policy = net(states)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
            records.extend([epoch_loss, epoch_acc])
            
        if epoch_acc > best_acc:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'records': np.array(records)
                }
            torch.save(checkpoint, os.path.join(model_path,'best_for_analysis.pt'))

            traced = torch.jit.trace(net.cpu(), torch.rand(1, 23, 32, 32))
            traced.save(os.path.join(model_path,'best.pth'))
            best_acc = epoch_acc

        
        if epoch % save_every == 0 or epoch + 1 == num_epochs:
            checkpoint = {
	        'epoch': epoch,
	        'state_dict': net.state_dict(),
	        'optimizer_state_dict': optimizer.state_dict(),
            'records': np.array(records)
            }
            torch.save(checkpoint, os.path.join(model_path,'{}.pt'.format(epoch)))
        
    return records

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using ', device)

    set_seed()

    learning_rate = config['lr']
    load_prop = config['load_prop']
    batch_size = config['batch_size']
    save_every = config['save_every']
    num_epochs = config['epoch']
    option = config['option']

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_path = 'model_checkpoints/{}'.format(st)
    os.makedirs(model_path)

    episode_dir = '../data/episodes'
    obses, samples = create_dataset_from_json(episode_dir, load_prop=load_prop)
    u_labels = [sample[-1] for sample in samples]
    
    u_actions = ['north', 'south', 'west', 'east', 'bcity']
    print('obses:', len(obses), 'samples:', len(samples))

    for value, count in zip(*np.unique(u_labels, return_counts=True)):
        print(f'{u_actions[value]:^5}: {count:>3}')

    print('option:', option, 'load prop:', load_prop)
    train, val = train_test_split(samples, test_size=0.1, random_state=42)
    train_loader = DataLoader(LuxDataset(obses, train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LuxDataset(obses, val), batch_size=batch_size, shuffle=False)

    net = Autoencoder(option=option).to(device)
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    records = train_model(net, model_path ,dataloaders_dict, criterion, optimizer,
        num_epochs=num_epochs, device=device, save_every=save_every)
    
    np.savetxt(os.path.join(model_path,'records.csv'), np.array(records), delimiter=",")
    with open(os.path.join(model_path,'config.json'), 'w') as f:
        json.dump(config, f)
    print('Work Done.')
