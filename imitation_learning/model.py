from torch import nn
import torch
import torch.nn.functional as F

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

class Autoencoder(nn.Module):
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

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self,img_ch=CHANNEL, num_action=5, dim=64, map_size=12):
        super(U_Net,self).__init__()
        
        self.map_size = map_size
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=dim)
        self.Conv2 = conv_block(ch_in=dim,ch_out=dim*2)
        self.Conv3 = conv_block(ch_in=dim*2,ch_out=dim*4)
        
        if map_size >=24:
            self.Conv4 = conv_block(ch_in=dim*4,ch_out=dim*4)
        # self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024,ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

            self.Up4 = up_conv(ch_in=dim*4,ch_out=dim*4)
            self.Up_conv4 = conv_block(ch_in=dim*8, ch_out=dim*4)
        
        self.Up3 = up_conv(ch_in=dim*4,ch_out=dim*2)
        self.Up_conv3 = conv_block(ch_in=dim*4, ch_out=dim*2)
        
        self.Up2 = up_conv(ch_in=dim*2,ch_out=dim)
        self.Up_conv2 = conv_block(ch_in=dim*2, ch_out=dim)

        # self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        
        self.linear_head = nn.Sequential(
                                    nn.Dropout(0.5),
                                    nn.Linear(dim, dim),
                                    nn.BatchNorm1d(dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(dim, num_action)
                                    )
    def encode(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4,d5),dim=1)
        
        # d5 = self.Up_conv5(d5)
        
        if self.map_size >= 24:
            x4 = self.Maxpool(x3)
            x4 = self.Conv4(x4)
            d4 = self.Up4(x4)
            d4 = torch.cat((x3,d4),dim=1)
            x3 = self.Up_conv4(d4)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d2 = (d2 * x[:,:1]).view(d2.size(0), d2.size(1), -1).sum(-1) 
        return d2

    def forward(self,x):
        
        d2 = self.encode(x)
        d2 = self.linear_head(d2)

        # d1 = self.Conv_1x1(d2)

        return d2



    
