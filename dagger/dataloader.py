from os import listdir
import json, os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# 12: TB > RL > Tigga > P > LD
# 16: TB > RL > P > LD > Tigga
# 24: TB > P > LD > RL > Tigga
# 32: TB > LD > P > RL > Tigga

#TEAM = ['Toad Brigade', 'RL is all you need', 'Love Deluxe', 'Tigga', 'perspective']
#TEAM = ['Toad Brigade', 'RL is all you need']
#TEAM = ['Toad Brigade']
#TEAM = ['RL is all you need']
def to_label(action):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
        unit_id = strs[1]
    elif strs[0] == 'bcity':
        label = 4
        unit_id = strs[1]
    else:
        label = None
    return unit_id, label


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True

def create_dataset(obses_list, map_size=24): 
    obses = {}
    samples = []

    for data in obses_list:

        eps_id = data['EpisodeId']
        #if data['info']['TeamNames'][index] not in TEAM or \

        for i in range(len(data['steps'])-1):
            #if data['steps'][i][index]['status'] == 'ACTIVE':
            if True:
                actions = data['steps'][i+1]['action']
                obs = data['steps'][i]['observation']
                
                if not actions:
                    continue

                if depleted_resources(obs): # if no resources left, dump rest steps
                    break

                obs['player'] = 0
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{eps_id}_{i}'
                obses[obs_id] = obs
                                
                for action in actions:
                    unit_id, label = action
                    if label is not None:
                        samples.append((obs_id, unit_id, label))

    return obses, samples

def make_input(obs, unit_id):

    width, height = obs['width'], obs['height']
    #x_shift = (24 - width) // 2
    #y_shift = (24 - height) // 2
    cities = {}
    
    b = np.zeros((21, width, width), dtype=np.float32)
    pos_x, pos_y = 0, 0
    count_u, count_ct = 0, 0

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        # example: 'u 0 0 u_11 2 26 0 100 0 0'
        # identifier unit_type team unit_id pos_x pos_y cooldown wood coal uranium
        if input_identifier == 'u': 
            #count_u += 1
            x = int(strs[4]) #+ x_shift
            y = int(strs[5]) #+ y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])

            if unit_id == strs[3]:
                pos_x, pos_y = x, y
                # Position and Cargo
                b[:2, x, y] = (1, (wood + coal + uranium) / 100) 
            #else:
                # Units
            team = int(strs[2])
            cooldown = float(strs[6])

            # idx 0:3 own resources
            # idx 3:6 adversary's resources
            #idx = (team - obs['player']) % 2 * 3 
            idx = 2 + (team - obs['player']) % 2 * 3 
            b[idx:idx + 3, x, y] = (1, cooldown / 6, (wood + coal + uranium) / 100)

        elif input_identifier == 'ct':
            # CityTiles
            #count_ct += 1
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) #+ x_shift
            y = int(strs[4]) #+ y_shift
            cooldown = float(strs[5])

            if unit_id == strs[3]+strs[4]:
                pos_x, pos_y = x, y
            # idx 6:9 own citytiles
            # idx 9:12 adversary citytiles
            #idx = 6 or 8 (cooldown) + (team - obs['player']) % 2 * 3
            idx = 8 + (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] =  (1, cities[city_id], cooldown / 6)
            #b[idx:idx + 2, x, y] =  (1, cities[city_id])

        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) #+ x_shift
            y = int(strs[3]) #+ y_shift
            amt = int(float(strs[4]))
            #b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
            b[{'wood': 14, 'coal': 15, 'uranium': 16}[r_type], x, y] = amt / 800

        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            #b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
            b[17 + (team - obs['player']) % 2, :] = min(rp, 200) / 200

        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10

        #elif input_identifier == 'ccd':
            # Road
        #    x = 
    # #unity, #citytile
    #b[17, :] = count_u / 100
    #b[18, :] = count_ct / 100

    # Day/Night Cycle
    b[19, :] = obs['step'] % 40 / 40
    # Turns
    b[20, :] = obs['step'] / 360
    
    # Map Size
    #b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1
    
    #distance_mask = np.zeros((32,32))
    #for i in range(len(distance_mask)):
    #    for j in range(len(distance_mask)):
    #        distance_mask[i, j] = 1 - ((i - pos_y)**2 + (j - pos_x)**2) / 2048
    #map_mask = np.zeros((32,32))
    #map_mask[x_shift:32 - x_shift, y_shift:32 - y_shift] = 1
    #b[22, :] = distance_mask
    #b[21, :] = map_mask

    return b


class LuxDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return states[idx], actions[idx]
