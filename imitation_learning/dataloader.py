from os import listdir
import json, os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


#TEAM = ['Toad Brigade', 'RL is all you need', 'Love Deluxe', 'Tigga', 'perspective']
TEAM = ['Toad Brigade', 'RL is all you need']
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

def create_dataset_from_json(episode_dir, load_prop=0.1): 
    obses = {}
    samples = []
    fpths = listdir(episode_dir)
    _, fpths = train_test_split(fpths,test_size=load_prop) if load_prop < 1 else (None, fpths)
    

    for fp in fpths:
        with open(os.path.join(episode_dir,fp), 'r') as f:
            data = json.load(f)

        eps_id = data['info']['EpisodeId']
        index = np.argmax([r or 0 for r in data['rewards']]) # winner index of current episode
        if data['info']['TeamNames'][index] not in TEAM:
            continue
        for i in range(len(data['steps'])-1):
            if data['steps'][i][index]['status'] == 'ACTIVE':
                actions = data['steps'][i+1][index]['action']
                obs = data['steps'][i][0]['observation']
                
                if not actions:
                    #print(fp,'step',i)
                    continue

                if depleted_resources(obs): # if no resources left, dump rest steps
                    break

                obs['player'] = index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{eps_id}_{i}'
                obses[obs_id] = obs
                                
                for action in actions:
                    unit_id, label = to_label(action)
                    if label is not None:
                        samples.append((obs_id, unit_id, label))

    return obses, samples

def make_input(obs, unit_id):
    '''
    --G1(target unit):-- (abandoned)
    --for current (x,y), --
    --idx 0: indicate whether if it's the targe unit(sub-agent)--
    --idx 1: all resources gained by target unit-- 

    G2(other unit):
    for current (x,y), 
    idx 2: indicate whether if it's a friendly unit
    idx 3: cooldown of this friendly unit
    idx 4: all resources gained by this friendly unit
    idx 5: indicate whether if it's an adversary unit
    idx 6: cooldown of this adversary unit
    idx 7: all resources gained by this adversary unit

    G3(citytile):
    for current (x,y), 
    idx 8: indicate whether if it's a friendly citytiles
    idx 9: longevity of this friendly citytiles (how long it can survive with current fuel)
    idx 10: indicate whether if it's an adversary citytiles
    idx 11: longevity of this adversary citytiles

    G4(resources):
    for current (x,y), 
    idx 12: wood
    idx 13: coal
    idx 14: uranium

    G5:
    for all (x,y), 
    idx 15: research point of own team
    idx 16: research point of adversary

    G6:
    for all (x,y), 
    idx 17: current cycle
    idx 18: current step
    --idx 19: indicate current (x,y) is valid or not--(abandoned)
    '''

    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((20, 32, 32), dtype=np.float32)
    pos_x, pos_y = 0, 0
    count_u, count_ct = 0, 0

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        # example: 'u 0 0 u_11 2 26 0 100 0 0'
        # identifier unit_type team unit_id pos_x pos_y cooldown wood coal uranium
        if input_identifier == 'u': 
            #count_u += 1
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
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
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            cooldown = float(strs[5])

            if unit_id == strs[3]+strs[4]:
                pos_x, pos_y = x, y
            # idx 6:9 own citytiles
            # idx 9:12 adversary citytiles
            #idx = 6 or 8 (cooldown) + (team - obs['player']) % 2 * 3
            idx = 8 + (team - obs['player']) % 2 * 2
            #b[idx:idx + 3, x, y] =  (1, cities[city_id], cooldown / 6)
            b[idx:idx + 2, x, y] =  (1, cities[city_id])

        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
            #b[{'wood': 14, 'coal': 15, 'uranium': 16}[r_type], x, y] = amt / 800

        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
            #b[17 + (team - obs['player']) % 2, :] = min(rp, 200) / 200

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
    b[17, :] = obs['step'] % 40 / 40
    # Turns
    b[18, :] = obs['step'] / 360
    
    # Map Size
    #b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1
    
    #distance_mask = np.zeros((32,32))
    #for i in range(len(distance_mask)):
    #    for j in range(len(distance_mask)):
    #        distance_mask[i, j] = 1 - ((i - pos_y)**2 + (j - pos_x)**2) / 2048
    map_mask = np.zeros((32,32))
    map_mask[x_shift:32 - x_shift, y_shift:32 - y_shift] = 1
    #b[22, :] = distance_mask
    b[19, :] = map_mask

    return b


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        #state, distance_mask, map_mask = make_input(obs, unit_id)
        state = make_input(obs, unit_id)
        return state, action