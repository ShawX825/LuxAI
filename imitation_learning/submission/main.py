import os, sys
import numpy as np
import torch
from lux.game import Game


path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
#model = torch.jit.load(f'{path}/best.pth')
#model.eval()
models = {}
for w in [12,16,24,32]:
    models[w] = torch.jit.load(f'{path}/{w}.pth')
    models[w].eval()

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
            b[{'wood': 14, 'coal': 15, 'uranium': 16}[r_type], x, y] = amt / 800

        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
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


game_state = None
def get_game_state(observation):
    global game_state
    
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation["player"]
    else:
        game_state._update(observation["updates"])
    return game_state


def in_city(pos):    
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False

def invalid_pos(pos):
    try:
        map_size = game_state.map.map_width
        city = game_state.map.get_cell_by_pos(pos).citytile
        return (city is not None and city.team != game_state.id) or \
        pos.x < 0 or pos.x >= map_size or pos.y < 0 or pos.y >= map_size
    except:
        return True



def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)

unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]
def get_action(policy, unit, dest):
    for label in np.argsort(policy)[::-1]:
        act = unit_actions[label]
        pos = unit.pos.translate(act[-1], 1) or unit.pos
        # if pos not in dest or in_city(pos)
        if ((label == 4 and unit.can_build(game_state.map)) or \
        pos not in dest or in_city(pos)) and not invalid_pos(pos,game_state):
            return call_func(unit, *act), pos 
            
    return unit.move('c'), unit.pos

def agent(observation, configuration):
    global game_state

    game_state = get_game_state(observation)    
    player = game_state.players[observation.player]
    actions = []
    width =  observation['width']
    model = models[width]
    
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
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            state = make_input(observation, unit.id)
            with torch.no_grad():
                p = model(torch.from_numpy(state).unsqueeze(0))

            policy = p.squeeze(0).numpy()

            action, pos = get_action(policy, unit, dest)
            actions.append(action)
            dest.append(pos)

    return actions