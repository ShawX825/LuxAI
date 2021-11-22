from model import *
from kaggle_environments import make
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os, time, datetime, json, pickle, copy
from lux.game import Game


CHANNEL = 21
difficulty = 1.0
SEED = 42
game_state = Game()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_reward(obs,alpha=1.0, clip=True, clip_val=1.0):
    global game_state
    score = lambda p: p.city_tile_count*3 + len(p.units) #+ p.research_points / 100
    player, opponent = game_state.players[obs["player"]], game_state.players[1-obs["player"]] 
    player_score, opponent_score =  score(player), score(opponent) #+ obs.step / 400
    reward = player_score - alpha*opponent_score
    return np.tanh(reward)*clip_val if clip else reward

def test(agent, o, config):
    map_size = config['map']
    env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size,\
     "loglevel": config['loglevel']},debug=config['debug'])
    win, tie = 0, 0
    for episode in range(config['test_eps']):
        env.reset()
        if np.random.rand(1) < 0.5: 
            env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size,\
             "loglevel": config['loglevel']},debug=config['debug'])
        steps = env.run([agent,o])
        #print(steps[-1][0]['reward'],steps[-1][1]['reward'])
        if steps[-1][0]['reward'] > steps[-1][1]['reward']:
            win += 1
        elif steps[-1][0]['reward'] == steps[-1][1]['reward']:
            tie += 1
    return win, tie

def train(config):
    
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print('using ', device)

    set_seed()

    learning_rate = config['lr']
    batch_size = config['batch_size']
    num_epochs = config['epoch']
    map_size = config['map']
    train_steps = config['train_steps']
    pre_trained = config['pre_trained']
    clip_val = config['clip_val']
    num_action = 5

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_path = 'model_checkpoints/{}_{}'.format(st,map_size)
    os.makedirs(model_path)
    with open(os.path.join(model_path,'config.json'), 'w') as f:
        json.dump(config, f)

    # loading preprocessed embeddings and labels
    with open('emb/embedding_{}.pkl'.format(map_size),'rb') as f:
        expert_em = pickle.load(f)
    with open('emb/labels_{}.pkl'.format(map_size),'rb') as f:
        expert_labels = pickle.load(f)   
    expert_em = torch.from_numpy(expert_em).to(device)
    expert_agent = Agent(device, map_size)

    option = {12:3,16:3,24:3,32:8}[map_size]
    model = ResNet(option=option).to(device)

    if pre_trained:
        pre_trained = '../imitation_learning/submission/IL1104/{}.pth'.format(map_size)
        pretrained_model = torch.jit.load(pre_trained)
        model.load_state_dict(pretrained_model.state_dict())
        depth = len(list(pretrained_model.parameters())) # including bias
        for i, param in enumerate(model.parameters()):
            if i < (depth - config['transfer'] * 2): # transfer learning on last 3 layers
                param.requires_grad = False

    # DAgger begins
    # line 1-2: Initialize D and Policy Net
    D = {'states':[], 'actions':[]}
    best_reward = -50
    # line 3
    env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size,\
                "loglevel": config['loglevel']}, debug=config['debug'])
    for episode in range(num_epochs):

        env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size,\
                "loglevel": config['loglevel']}, debug=config['debug'])

        # initial observation
        env.reset() 
        trainer = env.train([None, expert_agent])
        obs = trainer.reset()
        # update initial state
        # here, env.state[0].observation == obs
        # return of trainer.reset() or trainer.step() == "obs" of return of env.reset()
        game_state._initialize(env.state[0].observation["updates"])
        game_state._update(env.state[0].observation["updates"][2:])
        #player = game_state.players[env.state[0].observation.player]
        #opponent = game_state.players[(env.state[1].observation.player + 1) % 2]
        episode_reward = 0
        current_reward = get_reward(obs, difficulty, clip=False, clip_val=clip_val)
        beta = 0.5 ** episode

        # line 4-5: Sample T steps
        for t in range(360): # lux-ai has 360 turns
            # Model takes action
            unit_states, actions, labeled_actions, labels_exp = agent(obs, game_state, model,\
             beta, expert_em, expert_labels, num_action, device)

            # Apply the action to the environment
            next_obs, _, done, _ = trainer.step(actions)

            # update game state
            #game_state_copy = copy.deepcopy(game_state)
            game_state._update(next_obs["updates"])
            #next_game_state_copy = copy.deepcopy(game_state)
            obs_copy = copy.deepcopy(obs)
            next_obs_copy = copy.deepcopy(next_obs)

            # line 6-7: D = D U D_i
            D['states'].extend(unit_states) 
            D['actions'].extend(np.array(labels_exp))
            obs = next_obs
            
            next_reward = get_reward(obs, difficulty, clip=False, clip_val=clip_val)
            current_reward = next_reward 
            
            # one of the players don't have citytiles or units
            # or 360 steps finished
            if done: 
                episode_reward = current_reward
                break      
        
        # line 8: Train the model on D
        episode_losses = 0
        train_loader = DataLoader(LuxDataset(D['states'], D['actions']), batch_size=batch_size, shuffle=True,\
        num_workers=0, worker_init_fn=np.random.seed(SEED))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for i in range(train_steps):
            loss, acc = optimize(model, train_loader, criterion, optimizer, device, batch_size)
            episode_losses += loss

        print("[Episode {}]\tsteps: {},\tDataset Size: {},\trewards : {:.3f},\tavg loss: : {:.6f}, \tbeta : {:.1f}%"\
        .format(episode, t, len(D['states']), episode_reward, episode_losses/train_steps, beta*100, ))

        if episode_reward > best_reward:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, CHANNEL, map_size, map_size))
            traced.save(os.path.join(model_path,'{}.pth'.format(map_size)))
            best_reward = episode_reward
            model.to(device)

    # win-rate test
    print('Testing...')
    opponents = [expert_agent,"simple_agent"]
    if os.path.exists(os.path.join(model_path,'{}.pth'.format(map_size))):
        model = torch.jit.load(os.path.join(model_path,'{}.pth'.format(map_size))).to(device)

    dagger_agent = Agent(device,map_size,model=model)

    for i,o in enumerate(opponents):
        win, tie = test(dagger_agent, o, config)
        print("[Test {}]\tRound: {},\tWin: {},\tTie: {},\tWin Rate: {:.2f}%"\
        .format(i,config['test_eps'],win,tie,100*win/(config['test_eps']-tie)))
    

