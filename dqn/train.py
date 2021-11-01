from model import *
from kaggle_environments import make
import torch.nn.functional as F
import torch

CHANNEL = 21

def compute_loss(model, target, states, actions, rewards, next_states, dones):
    
    now_values = model(states).gather(1, actions)
    max_next_values = torch.max(target(next_states),1)[0].detach()[:,None]
    
    loss = F.smooth_l1_loss(now_values, rewards + (gamma * max_next_values)*(1-dones))
    
    return loss

def get_env(map_size, agent_teacher):
    env = make("lux_ai_2021", configuration={"width": map_size, "height": map_size})
    return env.train([None, agent_teacher])

def train(model_class, env, map_size):
    # Initialize model and target network
    model = model_class(input_shape=(CHANNEL,map_size,map_size), num_action=5).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            action = model.act(state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
        
        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.1:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
    return model
