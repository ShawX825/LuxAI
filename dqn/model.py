from torch import nn
import torch
import random
import torch.nn.functional as F
import collections
import math
import numpy as np
from lux.game import Game

CHANNEL = 21
Transition = collections.namedtuple('Transition', \
('game_state', 'obs','action', 'reward', 'next_game_state', 'next_obs', 'done'))
T = collections.namedtuple('T', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=10000):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def push(self, t):
        states, next_states, dones = [], [], []
        player = t.game_state.players[t.obs.player]
        next_player = t.next_game_state.players[t.next_obs.player]
        num = len(t.action)
        alive_units = [unit for unit in player.units if unit in next_player.units]
        for unit in player.units:
            states.append(make_input(t.obs, unit.id))
            if unit in next_player.units:
                next_states.append(make_input(t.next_obs, unit.id))
                dones.append(0)
            else:
                next_states.append(make_input(t.obs, unit.id))
                dones.append(1)

        transitions = list(zip(states,t.action,[t.reward]*num,next_states,dones))

        self.buffer.extend(transitions)
    
    def sample(self, batch_size, device):
        batch_trans = random.sample(self.buffer, batch_size)
        batch_trans = T(*zip(*batch_trans))

        states = torch.tensor(batch_trans.state, dtype=torch.float, device=device)
        actions = torch.tensor(batch_trans.action, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(batch_trans.reward, dtype=torch.float, device=device).unsqueeze(1)
        next_states = torch.tensor(batch_trans.next_state, dtype=torch.float, device=device)
        dones = torch.tensor(batch_trans.done, dtype=torch.float, device=device).unsqueeze(1)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        )
    def forward(self, x):
        return self.conv(x)   

def make_layers(in_shape, layers):
    skips = []
    convs = []
    c, w, _ = in_shape
    for layer in layers:
        in_channel, out_channel, kernel, stride, padding = layer

        w, old_w = (w - kernel + 2*padding) // stride + 1, w
        c, old_c = out_channel, c

        if in_channel > out_channel:
            skips.append(nn.ConvTranspose2d(in_channel,out_channel,1,2,padding=0,output_padding=1,bias=False))
        elif in_channel == out_channel and w == old_w:
            skips.append(nn.Identity())
        else:
            skips.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride))
        

        convs.append(BasicConvBlock(in_channel, out_channel, kernel, stride, padding) if in_channel <= out_channel \
            else BasicDeconvBlock(in_channel, out_channel, kernel, stride, padding))

    return nn.ModuleList(skips), nn.ModuleList(convs), c*w*w

opt = { 
    3: {'conv_layers': [[CHANNEL,64,3,1,1],[64,64,3,1,1],[64,128,3,1,1],\
        [128,128,3,1,1],[128,256,3,1,1],[256,256,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    } # best
}


class QNet(nn.Module):
    def __init__(self, input_shape=(CHANNEL,24,24), num_action=5, option=3):
        super().__init__()
        self.in_channel, self.in_w, self.in_h = input_shape
        self.conv_layers = opt[option]['conv_layers']
        self.hidden_shape = opt[option]['conv_layers'][-1][1]
        self.conv_skips, self.conv_blocks, self.in_feature = make_layers(input_shape, self.conv_layers)
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear_head = nn.Linear(self.hidden_shape, num_action)
        self.relu = opt[option]['nonlinearity']
    
    def encode(self, input):
        x = input
        for i in range(len(self.conv_layers)):
            x = self.relu(self.conv_skips[i](x) + self.conv_blocks[i](x))  
        x = (x * input[:,:1]).view(x.size(0), x.size(1), -1).sum(-1) 
        return x 

    def forward(self,input):
        x = self.encode(input)
        #x = self.dropout1(x)
        x = self.linear_head(x)
        return x


def make_input(obs, unit_id):

    width, height = obs['width'], obs['height']
    cities = {}
    
    b = np.zeros((21, width, width), dtype=np.float32)
    pos_x, pos_y = 0, 0
    count_u, count_ct = 0, 0

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        # example: 'u 0 0 u_11 2 26 0 100 0 0'
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

            idx = 2 + (team - obs['player']) % 2 * 3 
            b[idx:idx + 3, x, y] = (1, cooldown / 6, (wood + coal + uranium) / 100)

        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) #+ x_shift
            y = int(strs[4]) #+ y_shift
            cooldown = float(strs[5])

            if unit_id == strs[3]+strs[4]:
                pos_x, pos_y = x, y

            idx = 8 + (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] =  (1, cities[city_id], cooldown / 6)

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

    # Day/Night Cycle
    b[19, :] = obs['step'] % 40 / 40
    # Turns
    b[20, :] = obs['step'] / 360

    return b


def act(input, epsilon=0.0):
    if not isinstance(input, torch.FloatTensor):
        input = torch.from_numpy(input).float().unsqueeze(0).to(device)
    if random.random() < epsilon:
        return random.randint(0, self.num_actions-1)
    else:
        return torch.argmax(self.forward(input)).item()

def in_city(pos,game_state):    
    
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False


def invalid_pos(pos, game_state):
    try:
        map_size = game_state.map_width
        city = game_state.map.get_cell_by_pos(pos).citytile
        return (city is not None and city.team != game_state.id) or \
        pos.x < 0 or pos.x >= map_size or pos.y < 0 or pos.y >= map_size
    except:
        return True   
        
    


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)

unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]
def get_action(policy, unit, dest, width, valid, game_state):
    for label in np.argsort(policy)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos
        if valid:
            if label == 4 and unit.can_build(game_state.map):
                return call_func(unit, *act), pos, label 
            if (pos not in dest or in_city(pos,game_state)) and not invalid_pos(pos,game_state):
                return call_func(unit, *act), pos, label 
        else:
            if pos not in dest or in_city(pos,game_state):
                return call_func(unit, *act), pos, label           
            
    return unit.move('c'), unit.pos, label

def agent(observation, game_state, model, epsilon, num_action, device):
    
    #game_state = get_game_state(observation)    
    player = game_state.players[observation.player]
    actions, labels = [], []
    width =  observation['width']
    #model = models[width]
    
    # City Actions
    unit_count = len(player.units)
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                if unit_count < player.city_tile_count: 
                    actions.append(city_tile.build_worker())
                    unit_count += 1
                elif not player.researched_uranium():
                    actions.append(city_tile.research())
                    player.research_points += 1
    
    # Worker Actions
    dest = []
    e = np.random.rand(1) 
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos,game_state)):
            
            # with probability epsilon to select a random action (explore)
            if e < epsilon:
                policy = np.random.rand(num_action)

            else: # exploit
                state = make_input(observation, unit.id)
                with torch.no_grad():
                    p = model(torch.from_numpy(state).to(device).unsqueeze(0))

                policy = p.squeeze(0).detach().cpu().numpy()
            action, pos, label = get_action(policy, unit, dest, width, False, game_state)
            actions.append(action)
            dest.append(pos)
            labels.append(label)

    return actions, labels

def compute_loss(model, target, gamma, states, actions, rewards, next_states, dones):
    
    now_values = model(states).gather(1, actions)
    max_next_values = torch.max(target(next_states),1)[0].detach()[:,None]
    
    loss = F.smooth_l1_loss(now_values, rewards + (gamma * max_next_values)*(1-dones))
    
    return loss

def optimize(model, target, memory, optimizer, device, batch_size, gamma):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size, device) 
    loss = compute_loss(model, target, gamma, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode, max_epsilon,min_epsilon, epsilon_decay):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

class Agent():
    def __init__(self,device,map_size,valid=False,model=None,path='../imitation_learning/submission/IL1104'):
        self.model = torch.jit.load(f'{path}/{map_size}.pth') if model is None else model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.game_state = Game()
        self.valid = valid        

    def get_game_state(self, observation):
        if observation["step"] == 0:
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation["player"]
        else:
            self.game_state._update(observation["updates"])

    def __call__(self,observation,configuration):
        width =  observation['width']
        self.get_game_state(observation)    
        player = self.game_state.players[observation.player]
        actions = []
       # print("Agent:",self.game_state.id)
        #print("teacher!!!", observation.player)
        # City Actions
        unit_count = len(player.units)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                if city_tile.can_act():
                    if unit_count < player.city_tile_count: 
                        actions.append(city_tile.build_worker())
                        unit_count += 1
                    elif not player.researched_uranium():
                        actions.append(city_tile.research())
                        player.research_points += 1
        
        # Worker Actions
        dest = []
        for unit in player.units:
            if unit.can_act() and (self.game_state.turn % 40 < 30 or not in_city(unit.pos, self.game_state)):
                state = make_input(observation, unit.id)
                with torch.no_grad():
                    p = self.model(torch.from_numpy(state).to(self.device).unsqueeze(0))

                policy = p.squeeze(0).detach().cpu().numpy()
                action, pos, labels = get_action(policy, unit, dest, width, self.valid, self.game_state)
                actions.append(action)
                dest.append(pos)

        return actions