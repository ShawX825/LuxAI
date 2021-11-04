from model import *
from kaggle_environments import make
import torch.nn.functional as F
import numpy as np
import torch
from lux.game import Game
import time

CHANNEL = 21
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 2000
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_env(map_size):
    env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size})
    return env

def get_reward(game_state, obs,alpha=1.0):
    score = lambda p: p.city_tile_count*3 + len(p.units) + p.research_points / 100
    player, opponent = game_state.players[obs["player"]], game_state.players[1-obs["player"]] 
    player_score, opponent_score =  score(player), score(opponent) #+ obs.step / 400
    reward = player_score - alpha*opponent_score
    return reward

# game_state = Game()
def _train(game_state, map_size, agent_teacher, device, num_action=5):
    # Initialize model and target network
    model = QNet(input_shape=(CHANNEL,map_size,map_size), num_action=5).to(device)
    target = QNet(input_shape=(CHANNEL,map_size,map_size), num_action=5).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size})

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        # (1-e) chance to generate a new environment 
        # when at exploring stage, it's more likely to explore in the same environment 
        if np.random.rand(1) < 1 - epsilon: 
            env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size})
        # initial observation
        env.reset() 
        trainer = env.train([None, agent_teacher])
        obs = trainer.reset()
        # update initial state
        # here, env.state[0].observation == obs
        # return of trainer.reset() or trainer.step() == "obs" of return of env.reset()
        game_state._initialize(env.state[0].observation["updates"])
        game_state._update(env.state[0].observation["updates"][2:])
        #player = game_state.players[env.state[0].observation.player]
        #opponent = game_state.players[(env.state[1].observation.player + 1) % 2]
        
        current_reward = get_reward(game_state,obs)
        episode_reward = current_reward

        for t in range(360): # lux-ai has 360 turns
            # Model takes action
            actions = agent(obs, game_state, model, epsilon, num_action)

            # Apply the action to the environment
            next_obs, _, done, _ = env.step(action)

            # update game state
            game_state_copy = copy.deepcopy(game_state)
            game_state._update(next_obs["updates"])
            next_game_state_copy = copy.deepcopy(game_state)
            obs_copy = copy.deepcopy(obs)
            next_obs_copy = copy.deepcopy(next_obs)

            # dedicated reward function
            next_reward = get_reward(game_state, next_obs)

            # Save transition to replay buffer
            memory.push(Transition(game_state_copy, obs_copy, actions, \
            next_reward - current_reward, next_game_state_copy, next_obs_copy, done))
            
            obs = next_obs
            current_reward = next_reward 
        
            
            # one of the players don't have citytiles or units
            # or 360 steps finished
            if done: 
                episode_reward = current_reward
                break

        rewards.append(episode_reward)
        
        # Train the model if memory is sufficient
        episode_losses = 0
        if len(memory) > min_buffer:
            #if np.mean(rewards[print_interval:]) < 0.1:
            #    print('Bad initialization. Please restart the training.')
            #    exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer, device)
                episode_losses += loss.item()
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, episode_rewards, episode_losses/train_steps, len(memory), epsilon*100))

        #if episode % print_interval == 0 and episode > 0:
        #print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
        #                    episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
    return model

def train(config):

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print('using ', device)

    set_seed()

    '''
    learning_rate = config['lr']
    load_prop = config['load_prop']
    batch_size = config['batch_size']
    save_every = config['save_every']
    num_epochs = config['epoch']
    option = config['option']
    map_size = config['map']'''
    map_size = 12

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_path = 'model_checkpoints/{}'.format(st)
    os.makedirs(model_path)

    game_state = Game()
    _train(game_state, map_size, agent_teacher, device):