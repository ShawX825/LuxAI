from torch import nn
import torch.nn.functional as F

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

        skips.append(nn.ConvTranspose2d(in_channel,out_channel,1,2,padding=0,output_padding=1,bias=False) if in_channel > out_channel \
            else nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2))

        convs.append(BasicConvBlock(in_channel, out_channel, kernel, stride, padding) if in_channel < out_channel \
            else BasicDeconvBlock(in_channel, out_channel, kernel, stride, padding))
        
        w = (w - kernel + 2*padding) // stride + 1
        c = out_channel
        
    return nn.ModuleList(skips), nn.ModuleList(convs), c*w*w

class Autoencoder(nn.Module):
    def __init__(self, input_shape=(21,32,32), hidden_shape = 512):
        super().__init__()
        self.in_channel, self.in_w, self.in_h = input_shape
        self.conv_layers = [[21,64,3,2,1],[64,128,3,2,1],[128,256,3,2,1],[256,512,3,2,1]]
        self.conv_skips, self.conv_blocks, self.in_feature = make_layers(input_shape, self.conv_layers)
        self.deconv_layers = [[512,256,3,2,1],[256,128,3,2,1],[128,64,3,2,1],[64,21,3,2,1]]
        self.deconv_skips, self.deconv_blocks, _ = make_layers((512,2,2), self.deconv_layers)

        self.linear1 = nn.Linear(self.in_feature, hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, self.in_feature)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
    
    def encode(self, x):
        for i in range(len(self.conv_layers)):
            x = self.relu(self.conv_skips[i](x) + self.conv_blocks[i](x)) 
        x = self.flatten(x)
        x = self.linear1(x)
        return x 
    
    def decode(self, x):
        x = self.relu(x)
        x = self.relu(self.linear2(x))
        x = x.view(x.size(0),512,2,2)
        for i in range(len(self.deconv_layers)-1):
            x = self.relu(self.deconv_skips[i](x) + self.deconv_blocks[i](x))
        x = self.deconv_skips[-1](x) + self.deconv_blocks[-1](x)
        return x
    
    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x

        

