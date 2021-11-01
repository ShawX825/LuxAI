from torch import nn
import torch
import torch.nn.functional as F

CHANNEL = 21
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        
        batch_trans = random.sample(self.buffer, batch_size)
        
        batch_trans = Transition(*zip(*batch_trans))
        
        states = torch.tensor(batch_trans.state, dtype=torch.float, device=device)
        actions = torch.tensor(batch_trans.action, dtype=torch.long, device=device)
        rewards = torch.tensor(batch_trans.reward, dtype=torch.float, device=device)
        next_states = torch.tensor(batch_trans.next_state, dtype=torch.float, device=device)
        dones = torch.tensor(batch_trans.done, dtype=torch.float, device=device)
        
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

