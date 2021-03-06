from dataloader import *
from model import *
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os, time, datetime, json, pickle

SEED = 42
CHANNEL = 21

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_model(net, model_path, dataloaders_dict, criterion,
 optimizer, num_epochs, device, save_every, map_size):

    best_acc = 0.0
    best_loss = 2.0
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
            action_acc = [0]*5
            action_count = [0]*5
            
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
                    action_acc = [action_acc[i] + sum(torch.logical_and(preds == actions.data,preds==i)) for i in range(5)]
                    action_count = [action_count[i] + sum(actions.data == i) for i in range(5)]

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size
            action_acc = [action_acc[i] / action_count[i] for i in range(5)]
            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}'+
            f'| N: {action_acc[0]:.4f} | S: {action_acc[1]:.4f} | W: {action_acc[2]:.4f} | E: {action_acc[3]:.4f} |'+
            f'BC: {action_acc[4]:.4f}'
            )
                
            records.extend([epoch_loss, epoch_acc])
            
        if epoch_loss < best_loss:

            traced = torch.jit.trace(net.cpu(), torch.rand(1, CHANNEL, map_size, map_size))
            traced.save(os.path.join(model_path,'best_loss.pth'))
            best_loss = epoch_loss

        if epoch_acc > best_acc:

            traced = torch.jit.trace(net.cpu(), torch.rand(1, CHANNEL, map_size, map_size))
            traced.save(os.path.join(model_path,'best_acc.pth'))
            best_acc = epoch_acc
        
        if epoch % save_every == 0 or epoch + 1 == num_epochs:
            checkpoint = {
	        'epoch': epoch,
	        'state_dict': net.state_dict(),
	        'optimizer_state_dict': optimizer.state_dict(),
            'records': records
            }
            torch.save(checkpoint, os.path.join(model_path,'{}.pt'.format(epoch)))
        
    return records

def train(config):

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print('using ', device)

    set_seed()

    learning_rate = config['lr']
    load_prop = config['load_prop']
    batch_size = config['batch_size']
    save_every = config['save_every']
    num_epochs = config['epoch']
    option = config['option']
    map_size = config['map']

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_path = 'model_checkpoints/{}'.format(st)
    os.makedirs(model_path)

    episode_dir = '../data/episodes_top5' if config['mode'] != 'test' else '../data/episodes_test1'
    obses, samples = create_dataset_from_json(episode_dir, load_prop=load_prop, map_size=map_size)
    u_labels = [sample[-1] for sample in samples]
    
    u_actions = ['north', 'south', 'west', 'east', 'bcity']
    print('obses:', len(obses), 'samples:', len(samples))

    for value, count in zip(*np.unique(u_labels, return_counts=True)):
        print(f'{u_actions[value]:^5}: {count:>3}')
    
    print('option:', option, 'load prop:', load_prop)
    train, val = train_test_split(samples, test_size=0.1, random_state=42)
    train_loader = DataLoader(LuxDataset(obses, train, map_size), batch_size=batch_size, shuffle=True,\
        num_workers=0, worker_init_fn=np.random.seed(SEED))
    val_loader = DataLoader(LuxDataset(obses, val, map_size), batch_size=batch_size, shuffle=False,\
        num_workers=0, worker_init_fn=np.random.seed(SEED))

    net = Autoencoder(hidden_shape=1024, input_shape=(CHANNEL,map_size,map_size),option=option).to(device)
    #net = LuxNet().to(device)
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    time_start = time.time()
    records = train_model(net, model_path ,dataloaders_dict, criterion, optimizer,
        num_epochs=num_epochs, device=device, save_every=save_every, map_size=map_size)
    time_end = time.time()
    
    #np.savetxt(os.path.join(model_path,'records.csv'), np.array(records), delimiter=",")
    with open(os.path.join(model_path,'config.json'), 'w') as f:
        json.dump(config, f)
    print('Work Done. Total cost:',time_end - time_start, 's')
