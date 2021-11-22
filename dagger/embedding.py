from model import *
from dataloader import *
import numpy as np
from lux.game import Game
import time, json, pickle
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from os import listdir
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--map',type=int, default=32)

config = parser.parse_args()
map_size = vars(config)['map']
device = vars(config)['device']
dir_opt = {12:3,16:3,24:3,32:8}

episode_dir = '../data/episodes_top5' 
obses, samples = create_dataset_from_json(episode_dir, load_prop=1.0, map_size=map_size)
u_labels = [sample[-1] for sample in samples]
    
u_actions = ['north', 'south', 'west', 'east', 'bcity']
print('obses:', len(obses), 'samples:', len(samples))

for value, count in zip(*np.unique(u_labels, return_counts=True)):
    print(f'{u_actions[value]:^5}: {count:>3}')

path = '../imitation_learning/model_checkpoints/IL_OPT{}_{}/best_acc.pth'.format(dir_opt[map_size],map_size)
_model = torch.jit.load(path) 
model = Autoencoder(option=dir_opt[map_size])
model.load_state_dict(_model.state_dict())
model.to(device)
model.eval()
dataloader = DataLoader(LuxDataset(obses, samples, map_size), batch_size=128, shuffle=False)

em, actions = [], []

for item in dataloader:
    states, labels = item
    states = states.to(device).float()
    #states = states * distance_m * map_m
    embeddings = model.encode(states).detach().cpu().numpy()
    em.extend(embeddings)
    actions.extend(labels)

em = np.stack(em)
actions = np.stack(actions)
with open('embedding_{}.pkl'.format(map_size),'wb') as f:
    pickle.dump(em,f)

with open('labels_{}.pkl'.format(map_size),'wb') as f:
    pickle.dump(actions,f)

