from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lux.game import Game
import numpy as np
CHANNEL = 21

class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h

class LuxNet(nn.Module):
    def __init__(self): 
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(CHANNEL, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 5, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h)) # skip path
        # x[:,:1] is the mask, positions of target unit
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1) 
        p = self.head_p(h_head)
        return p

class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(  
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        )
        #self.bn = nn.BatchNorm2d(out_channel) if bn else None

    def forward(self, x):
        return self.conv(x)

class BasicDeconvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.deconv = nn.Sequential(  
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding=padding, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=1) 
        )

    def forward(self, x):
        return self.deconv(x)    

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
    0: {'conv_layers': [[CHANNEL,64,3,1,1]] + [[64,64,3,1,1] for _ in range(4)],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.ReLU()
    },
    1: {'conv_layers': [[CHANNEL,64,3,1,1]] + [[64,64,3,1,1] for _ in range(5)],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    2: {'conv_layers': [[CHANNEL,64,3,1,1]] + [[64,64,3,1,1] for _ in range(6)],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.ReLU()
    },
    3: {'conv_layers': [[CHANNEL,64,3,1,1],[64,64,3,1,1],[64,128,3,1,1],\
        [128,128,3,1,1],[128,256,3,1,1],[256,256,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    }, # best
    4: {'conv_layers': [[CHANNEL,64,3,1,1],[64,64,3,1,1],[64,128,3,1,1],\
        [128,128,3,1,1],[128,256,3,1,1],[256,256,3,1,1],[256,512,3,1,1],[512,512,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    5: {'conv_layers': [[CHANNEL,32,3,1,1],[32,64,3,1,1],[64,128,3,1,1],\
        [128,256,3,1,1],[256,512,3,1,1],[512,1024,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    6: {'conv_layers': [[CHANNEL,64,3,1,1],[64,128,3,1,1],[128,256,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    7: {'conv_layers': [[CHANNEL,32,3,1,1],[32,64,3,1,1],[64,128,3,1,1],[128,256,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    8: {'conv_layers': [[CHANNEL,64,3,1,1],[64,64,3,1,1],[64,128,3,1,1],[128,256,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    9: {'conv_layers': [[CHANNEL,64,3,1,1],[64,128,3,1,1],[128,128,3,1,1],[128,256,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    10: {'conv_layers': [[CHANNEL,64,3,1,1],[64,128,3,1,1],[128,256,3,1,1],[256,256,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    },
    11: {'conv_layers': [[CHANNEL,64,3,1,1],[64,64,3,1,1],[64,128,3,1,1],[128,128,3,1,1]],
        'conv_1': nn.Conv2d(64,128,3,2,1),
        'avg_pool': nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
        'dem': 8,
        'nonlinearity': nn.LeakyReLU()
    }

}

class ResNet(nn.Module):
    def __init__(self, input_shape=(CHANNEL,24,24), hidden_shape=512, num_action=5, option=0):
        super().__init__()
        self.in_channel, self.in_w, self.in_h = input_shape
        self.conv_layers = opt[option]['conv_layers']
        self.hidden_shape = opt[option]['conv_layers'][-1][1]
        self.conv_skips, self.conv_blocks, self.in_feature = make_layers(input_shape, self.conv_layers)
        #self.conv0 = nn.Conv2d(self.in_channel,64,3,1,1)
        #self.conv1 = opt[option]['conv_1']
        #self.avg_pool = opt[option]['avg_pool']
        #self.linear1 = nn.Linear(self.in_feature//opt[option]['dem'], hidden_shape)
        #self.linear2 = nn.Linear(hidden_shape, self.in_feature)
        self.dropout1 = nn.Dropout(p=0.5)
        #self.dropout2 = nn.Dropout(p=0.3)
        self.linear_head = nn.Linear(self.hidden_shape, num_action)
        #self.flatten = nn.Flatten()
        self.relu = opt[option]['nonlinearity']
    
    def encode(self, input):
        #x, distance_m, map_m = input[:,:22], input[:,22], input[:,21]
        #distance_m = distance_m.unsqueeze(1)
        #map_m = map_m.unsqueeze(1)
        #x = self.relu(self.conv0(x))
        x = input
        for i in range(len(self.conv_layers)):
            x = self.relu(self.conv_skips[i](x) + self.conv_blocks[i](x))
        
        x = (x * input[:,:1]).view(x.size(0), x.size(1), -1).sum(-1) 
        #x = x * distance_m * map_m
        #x = self.relu(self.conv1(x))
        #x = self.avg_pool(x)
        #x = self.flatten(x)
        #x = self.dropout2(x)
        #x = self.linear1(x)
        return x 
    
    #def decode(self, x):
    #    x = self.relu(x)
    #    x = self.relu(self.linear2(x))
    #    x = x.view(x.size(0),512,2,2)
    #    for i in range(len(self.deconv_layers)-1):
    #        x = self.relu(self.deconv_skips[i](x) + self.deconv_blocks[i](x))
    #    x = self.deconv_skips[-1](x) + self.deconv_blocks[-1](x)
    #    return x

    def forward(self,input):
        #x = self.relu(self.encode(input))
        x = self.encode(input)
        x = self.dropout1(x)
        x = self.linear_head(x)
        return x

def optimize(net, train_loader, criterion, optimizer, device, batch_size):
    epoch_loss, epoch_acc = 0, 0
    for item in train_loader:
        states, actions = item
        states = states.to(device).float()
        actions = actions.to(device).long()               
        optimizer.zero_grad()

        policy = net(states)
        loss = criterion(policy, actions)
        _, preds = torch.max(policy, 1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(policy)
        epoch_acc += torch.sum(preds == actions.data)
        
    data_size = len(train_loader.dataset)
    epoch_loss = epoch_loss / data_size
    epoch_acc = epoch_acc.double() / data_size
    return epoch_loss, epoch_acc

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


def in_city(pos,game_state):    
    
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False




def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)

unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]
def get_action(policy, unit, dest, width, game_state):
    for label in np.argsort(policy)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos
        if pos not in dest or in_city(pos,game_state):
            return call_func(unit, *act), pos, label           
            
    return unit.move('c'), unit.pos, label

def agent(observation, game_state, model, beta, expert_em, expert_labels, num_action, device):
    
    #game_state = get_game_state(observation)    
    player = game_state.players[observation.player]
    states, actions, labels = [], [], []
    actions_exp, labels_exp = [], []
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
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos,game_state)):
            state = make_input(observation, unit.id)
            with torch.no_grad():
                em = model.encode(torch.from_numpy(state).to(device).unsqueeze(0))
                p_classifier = F.softmax(model.linear_head(em).squeeze(0))
                errors = torch.sum((expert_em - em)**2, axis=1)
                p_expert = F.one_hot(torch.from_numpy(np.array(expert_labels[torch.argmin(errors)])),num_action).to(device)
                
                p = p_expert * beta + p_classifier * (1-beta)
                


            policy = p.detach().cpu().numpy()
            action, pos, label = get_action(policy, unit, dest, width, game_state)
            actions.append(action)
            dest.append(pos)
            labels.append(label)
            labels_exp.append(expert_labels[torch.argmin(errors)])
            states.append(state)

    return states, actions, labels, labels_exp

class LuxDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class Agent():
    def __init__(self,device,map_size,model=None,path='../imitation_learning/submission/IL1104'):
        self.model = torch.jit.load(f'{path}/{map_size}.pth') if model is None else model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.game_state = Game()

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
                action, pos, labels = get_action(policy, unit, dest, width,  self.game_state)
                actions.append(action)
                dest.append(pos)

        return actions