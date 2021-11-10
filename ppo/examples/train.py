import argparse
import glob
import os
import random
from typing import Callable
import gym

from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent_policy import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

import torch
import torch.nn as nn
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment 
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float) Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: (float)
        :return: (float) current learning rate
        """
        return progress_remaining * initial_value

    return func


def get_command_line_arguments():
    """
    Get the command line arguments
    :return:(ArgumentParser) The command line arguments as an ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Training script for Lux RL agent.')
    parser.add_argument('--id', help='Identifier of this run', type=str, default=str(random.randint(0, 10000)))
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--gamma', help='Gamma', type=float, default=0.995)
    parser.add_argument('--gae_lambda', help='GAE Lambda', type=float, default=0.95)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=512)  # 64
    parser.add_argument('--step_count', help='Total number of steps to train', type=int, default=10000000)
    parser.add_argument('--n_steps', help='Number of experiences to gather before each learning period', type=int, default=2048*8)
    parser.add_argument('--path', help='Path to a checkpoint to load to resume training', type=str, default=None)
    args = parser.parse_args()

    return args


class ResidualBlock(nn.Module):
    #实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,stride,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),#inplace = True原地操作
            nn.Conv2d(out_ch,out_ch,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_ch)
            )
        self.right = shortcut
        
    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return nn.ReLU()(out)

class ResNet(nn.Module):#224x224x3
    #实现主module:ResNet34
    def __init__(self, dim=64, in_chan=25, num_classes=1):
        super(ResNet,self).__init__()
        self.pre = nn.Sequential(
                nn.Conv2d(in_chan,dim,3,stride=1,padding=1,bias=False),# (224+2*p-)/2(向下取整)+1，size减半->112
                nn.ReLU(inplace = True),
                # nn.MaxPool2d(3,2,1)#kernel_size=3, stride=2, padding=1
                )#56x56x64
        
        #重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self.make_layer(dim,dim,2)#56x56x64,layer1层输入输出一样，make_layer里，应该不用对shortcut进行处理，但是为了统一操作。。。
        self.layer2 = self.make_layer(dim,dim*2,2,stride=1)#第一个stride=2,剩下3个stride=1;28x28x128
        self.layer3 = self.make_layer(dim*2,dim*4,3,stride=1)#14x14x256
        # self.layer4 = self.make_layer(256,512,3,stride=2)#7x7x512
        #分类用的全连接
        # self.fc = nn.Linear(512,num_classes)
        
    def make_layer(self,in_ch,out_ch,block_num,stride=1):
        #当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(#首个ResidualBlock需要进行option B处理
                nn.Conv2d(in_ch,out_ch,1,stride,bias=False),#1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch)
                )
        layers = []
        layers.append(ResidualBlock(in_ch,out_ch,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(out_ch,out_ch))#后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)
        
    def forward(self,x):    #224x224x3
        
        x = self.pre(x)     #56x56x64
        x = self.layer1(x)  #56x56x64
        x = self.layer2(x)  #28x28x128
        x = self.layer3(x)  #14x14x256
        # x = self.layer4(x)  #7x7x512
        # x = F.avg_pool2d(x,7)#1x1x512
        # x = x.view(x.size(0),-1)#将输出拉伸为一行：1x512
        # x = self.fc(x)    #1x1
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        return x

class resmlp(nn.Module):
    def __init__(self, input_size, dim):
        super(resmlp, self).__init__()
        
        self.model = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(input_size, dim),
                                nn.ReLU(),
                                nn.Linear(dim, dim),
                                nn.ReLU(),
                                nn.Linear(dim, input_size),
                                )
    
    def forward(self, x):
        return self.model(x)
    
    
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        input_size = 3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3
        dim = 512
        
        cnn_dim = 64
        self.layer1 = nn.Sequential(
                                nn.Linear(input_size, dim),
                                nn.ReLU(),
                                nn.Linear(dim, dim),
                                )
        # self.res1 = resmlp(input_size, dim)
        # self.res2 = resmlp(input_size, dim)
        # self.res3 = resmlp(input_size, dim)
        
        self.cnn = ResNet(dim = cnn_dim)
        
        self._features_dim = dim + cnn_dim*4
        # self._features_dim = cnn_dim*4

    def forward(self, observations) -> torch.Tensor:
        
        x = observations['unit_info']
        
        # for i in range(len(self.extractor)):
        #     x = self.extractor[i](x)
        
        x = self.layer1(x)
        # x = self.res1(x) + x
        # x = self.res2(x) + x
        # x = self.res3(x) + x
        
        # x = nn.Flatten()(x)
        
        # x = x + observations['unit_info']
        
        # print('x.shape:', x.shape)
        
        height, width = observations["size"][0].int()
        observations["map"] = observations["map"][:, :, :height, :width]
        
        # pos_x, pos_y = observations["pos"][0].int()
        
        # print('pos: {}, {}'.format(pos_x, pos_y))
        
        cnn_feat = self.cnn(observations['map'])
        
        # print('cnn_feat.shape: {}, obs.shape:{}'.format(cnn_feat.shape, observations['map'].shape))
        
        cnn_feat = (cnn_feat * observations["map"][:,15:16]).view(cnn_feat.size(0), cnn_feat.size(1), -1).sum(-1)
        
        # cnn_feat = cnn_feat[:, :, pos_x, pos_y]
        
        # cnn_feat = nn.Flatten()(nn.AvgPool2d(cnn_feat.shape[-1])(cnn_feat))
        
        output = torch.cat([x, cnn_feat], 1)
        
        # output = x
        
        # output = cnn_feat
        
        return output
    

def train(args):
    """
    The main training loop
    :param args: (ArgumentParser) The command line arguments
    """
    print(args)

    # Run a training job
    configs = LuxMatchConfigs_Default

    # Create a default opponent agent
    opponent = Agent()

    # Create a RL agent in training mode
    player = AgentPolicy(mode="train")

    # Train the model
    num_cpu = 1
    if num_cpu == 1:
        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)
    else:
        env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agent=opponent), i) for i in range(num_cpu)])
    run_id = args.id
    print("Run id %s" % run_id)

    if args.path:
        # by default previous model params are used (lr, batch size, gamma...)
        model = PPO.load(args.path)
        model.set_env(env=env)

        # Update the learning rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)

        # TODO: Update other training parameters
    else:
        policy_kwargs = dict(
                             features_extractor_class=CustomCombinedExtractor,
                             net_arch=[dict(pi=[512, 512], vf=[512, 512])]
                             )
        model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./lux_tensorboard/",
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    batch_size=args.batch_size,
                    n_steps=args.n_steps,
                    policy_kwargs=policy_kwargs,
                    )

    print("Training model...")
    # Save a checkpoint every 1M steps
    checkpoint_callback = EvalCallback(env, eval_freq = args.n_steps, n_eval_episodes = 20,
                                       best_model_save_path=f'./models/{run_id}_initial',)
    model.learn(total_timesteps=args.step_count,
                callback=checkpoint_callback)  # 20M steps
    if not os.path.exists(f'models/rl_model_{run_id}_{args.step_count}_steps.zip'):
        model.save(path=f'models/rl_model_{run_id}_{args.step_count}_steps.zip')
    print("Done training model.")

    # Inference the model
    # print("Inference model policy with rendering...")
    # saves = glob.glob(f'models/rl_model_{run_id}_*_steps.zip')
    # latest_save = sorted(saves, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
    # model.load(path=latest_save)
    # obs = env.reset()
    # for i in range(600):
    #     action_code, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, done, info = env.step(action_code)
    #     if i % 5 == 0:
    #         print("Turn %i" % i)
    #         env.render()

    #     if done:
    #         print("Episode done, resetting.")
    #         obs = env.reset()
    # print("Done")

    '''
    # Learn with self-play against the learned model as an opponent now
    print("Training model with self-play against last version of model...")
    player = AgentPolicy(mode="train")
    opponent = AgentPolicy(mode="inference", model=model)
    env = LuxEnvironment(configs, player, opponent)
    model = PPO("MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./lux_tensorboard/",
        learning_rate = 0.0003,
        gamma=0.999,
        gae_lambda = 0.95
    )

    model.learn(total_timesteps=2000)
    env.close()
    print("Done")
    '''
    
    
    print("Training model with self-play against last version of model...")
    
    model.learning_rate = 1e-4
    checkpoint_callback = EvalCallback(best_model_save_path=f'./models/{run_id}_self',)
    
    for i in range(30):
        print('Self Training round {}'.format(i))
        player = AgentPolicy(mode="train")
        opponent = AgentPolicy(mode="inference", model=model)
        env = LuxEnvironment(configs, player, opponent)
        # model = PPO("MlpPolicy",
        #     env,
        #     verbose=1,
        #     tensorboard_log="./lux_tensorboard/",
        #     learning_rate = 0.001,
        #     gamma=0.999,
        #     gae_lambda = 0.95
        # )
        model.set_env(env=env)

        # try:
        model.learn(total_timesteps=500000,
                    callback=checkpoint_callback)
        # except:
        #     print('skip errors')
        env.close()
    
        model.save(path=f'models/rl_self_model_{run_id}_{args.step_count}_steps.zip')
        
    print("Done")

if __name__ == "__main__":
    # Get the command line arguments
    local_args = get_command_line_arguments()

    # Train the model
    train(local_args)
