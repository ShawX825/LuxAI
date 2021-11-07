from model import *
from kaggle_environments import make
import torch.nn.functional as F
import numpy as np
import torch
from lux.game import Game
import time, datetime, os, copy
from teacher_agent.teacher import teacher

CHANNEL = 21
gamma         = 0.98
min_buffer    = 1000
target_update = 20 # episode(s)
max_epsilon   = 1.0
min_epsilon   = 0.01
print_interval= 20
game_state = Game()
teacher_model = None

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
    score = lambda p: p.city_tile_count*3 + len(p.units) + p.research_points / 100
    player, opponent = game_state.players[obs["player"]], game_state.players[1-obs["player"]] 
    player_score, opponent_score =  score(player), score(opponent) #+ obs.step / 400
    reward = player_score - alpha*opponent_score
    return np.tanh(reward)*clip_val if clip else reward

'''
def agent_teacher(observation,configuration):
    global game_state, teacher_model
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation["player"]
        print("teacher",game_state.id)
    else:
        game_state._update(observation["updates"])
    player = game_state.players[observation.player]
    print("teacher:", observation.player)
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
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            state = make_input(observation, unit.id)
            with torch.no_grad():
                p = teacher_model(torch.from_numpy(state).unsqueeze(0))

            policy = p.squeeze(0).numpy()
            action, pos, labels = get_action(policy, unit, dest)
            actions.append(action)
            dest.append(pos)

    return actions'''


def _train(target, model, model_path, agent_teacher, config,  num_action=5):
    global game_state
    # Initial settings

    difficulty = config['difficulty']
    map_size = config['map']
    device = config['device']
    max_episodes = config['max_eps']
    clip_val = config['clip_val']
    epsilon_decay = max_episodes // config['eps_decay']
    buffer_limit  = config['buffer_limit']
    train_steps = config['train_steps']
    batch_size = config['batch_size']
    learning_rate = config['lr']

    # Initialize replay buffer
    memory = ReplayBuffer(buffer_limit)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    survivals = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size,\
             "loglevel": config['loglevel']}, debug=config['debug'])
    new_env = 'new env'

    best_reward = 0
    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode, max_epsilon,min_epsilon, epsilon_decay)
        # (1-e) chance to generate a new environment 
        # when at exploring stage, it's more likely to explore in the same environment 
        '''
        if np.random.rand(1) < 1 - 1.5*epsilon: 
            env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size,\
             "loglevel": config['loglevel']}, debug=config['debug'])
            new_env = 'new env'
        else:
            new_env = 'old env'
            '''
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
        
        current_reward = get_reward(obs, difficulty, clip_val=clip_val)
        episode_reward = current_reward

        for t in range(360): # lux-ai has 360 turns
            # Model takes action
            actions, labeled_actions = agent(obs, game_state, model, epsilon, num_action, device)

            # Apply the action to the environment
            next_obs, _, done, _ = trainer.step(actions)

            # update game state
            game_state_copy = copy.deepcopy(game_state)
            game_state._update(next_obs["updates"])
            next_game_state_copy = copy.deepcopy(game_state)
            obs_copy = copy.deepcopy(obs)
            next_obs_copy = copy.deepcopy(next_obs)

            # dedicated reward function
            next_reward = get_reward(next_obs, difficulty, clip_val=clip_val)

            # Save transition to replay buffer
            memory.push(Transition(game_state_copy, obs_copy, labeled_actions, \
            next_reward - current_reward, next_game_state_copy, next_obs_copy, done))
            
            obs = next_obs
            current_reward = next_reward 
            
            # one of the players don't have citytiles or units
            # or 360 steps finished
            if done: 
                #print(t)
                episode_reward = current_reward
                break

        rewards.append(episode_reward)
        survivals.append(t)
        
        # Train the model if memory is sufficient
        episode_losses = 0
        if len(memory) > min_buffer:
            #if np.mean(rewards[20:]) < -3:
            #    print('Bad initialization. Please restart the training.')
            #    exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer, device, batch_size,  gamma)
                episode_losses += loss.item()
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        print("[Episode {}]\t{}\tsteps: {}\trewards : {:.3f},\tloss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%"\
        .format(episode, new_env, t, episode_reward, episode_losses/train_steps, len(memory), epsilon*100))

        if (episode % print_interval == 0 and episode > 0) or episode == max_episodes -1:
            print("[Last {} episodes]\tavg survive: {:.1f} steps,\tavg rewards: {:.3f},\tavg loss: : {:.6f}".format(
                    print_interval, np.mean(survivals[-1*print_interval:]), np.mean(rewards[-1*print_interval:]), np.mean(losses[print_interval*-20:])))

        if (episode % print_interval == 0 and episode > 0) and \
        np.mean(rewards[-1*print_interval:]) > best_reward:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, CHANNEL, map_size, map_size))
            traced.save(os.path.join(model_path,'best.pth'))
            best_reward = np.mean(rewards[-1*print_interval:])
            model.to(device)

        if episode == max_episodes - 1 or (episode % print_interval == 0 \
        and episode > 0 and np.mean(rewards[-1*print_interval:]) > 0):
            traced = torch.jit.trace(model.cpu(), torch.rand(1, CHANNEL, map_size, map_size))
            traced.save(os.path.join(model_path,'{}.pth'.format(episode)))
            model.to(device)


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
    global teacher_model

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    config['device'] = device
    print('using ', device)

    set_seed()
    pre_trained = '../imitation_learning/model_checkpoints/IL_OPT3_12/best_acc.pth'

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_path = 'model_checkpoints/{}'.format(st)
    os.makedirs(model_path)

    # Initialize model and target network
    map_size = config['map']
    model = QNet(input_shape=(CHANNEL,map_size,map_size), num_action=5).to(device)
    target = QNet(input_shape=(CHANNEL,map_size,map_size), num_action=5).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    if pre_trained:
        pretrained_model = torch.jit.load(pre_trained)
        model.load_state_dict(pretrained_model.state_dict())
        target.load_state_dict(pretrained_model.state_dict())
        target.eval()
        for i, param in enumerate(model.parameters()):
            if i < 26: # transfer learning on last 3 layers
                param.requires_grad = False
    
    #path = '../imitation_learning/submission/IL1101_1'
    #teacher_model = torch.jit.load(f'{path}/{map_size}.pth').to(device)

    agent_teacher_ = Agent(device,config['map'])
    _train(target, model, model_path, agent_teacher_, config=config)

    # test
    print('Testing...')
    opponents = [agent_teacher_,"simple_agent"]
    if os.path.exists(os.path.join(model_path,'best.pth')):
        model = torch.jit.load(os.path.join(model_path,'best.pth')).to(device)

    agent = Agent(device,config['map'],model=model)

    for i,o in enumerate(opponents):
        win, tie = test(agent, o, config)
        print("[Test {}]\tRound: {},\tWin: {},\tTie: {},\tWin Rate: {:.2f}%"\
        .format(i,config['test_eps'],win,tie,100*win/(config['test_eps']-tie)))