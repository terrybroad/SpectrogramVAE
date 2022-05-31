import torch
from torch import nn
from functools import reduce
from operator import __add__

def get_active_func(s):
    if s == "ReLU":
        return nn.ReLU()
    elif s == "LeakyReLU":
        return nn.LeakyReLU()
    elif s == "Tanh":
        return nn.Tanh()
    else:
        print("activation function %s, is not known, defaulting to LeakyReLU."%(s))
        return nn.LeakyReLU()
    

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.convs = nn.ModuleList()
        self.fc = nn.Linear(67584, self.latent_dim)
        self.sigmoid = nn.Sigmoid()
        
        hidden_dims = [64, 128, 256, 512, 512]

        in_channels = 1
        for h_dim in hidden_dims:
            self.convs.append(
                ConvLayer(in_channels,h_dim,(5,5),(2,2),activation_function="ReLU") 
            )
            in_channels = h_dim
    
    def reparameterisation(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def kl_divergence(self, mu, logvar):
        return  torch.mean((-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / self.latent_dim )

    def encode(self, input):
        x = input
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x, x

    def forward(self, input):
        mu, logvar = self.encode(input)
        kld = self.kl_divergence(mu, logvar)
        z = self.reparameterisation(mu, logvar)
        return z, kld, mu

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim
    ):
        super().__init__()
        h_dims = [512, 256, 128, 64]
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, 67584) #34816 for 256, 67584 for 512
        self.convs = nn.ModuleList()
        in_channels = h_dims[0]
        #Very unsatisfactory way of matching dimensions in encoder :/
        self.convs.append( 
            ConvLayer(
                in_channels,
                h_dims[0],
                (5,5),
                (1,2),
                padding=(0,2),
                activation_function="ReLU",
                transpose=True) )
        self.convs.append( 
            ConvLayer(
                in_channels,
                h_dims[1],
                (3,5),
                (2,2),
                padding=(0,2),
                activation_function="ReLU",
                transpose=True) )
        self.convs.append( 
            ConvLayer(
                h_dims[1],
                h_dims[2],
                (4,5),
                (2,2),
                padding=(2,2),
                activation_function="ReLU",
                transpose=True) )
        self.convs.append( 
            ConvLayer(
                h_dims[2],
                h_dims[3],
                (4,5),
                (2,2),
                padding=(1,2),
                activation_function="ReLU",
                transpose=True) )
        self.final_layer = ConvLayer(h_dims[3], 1, (4,5), (2,2), padding=(1,2), activation_function="Tanh",transpose=True) 

    def forward(self, input):
        x = self.decoder_input(input)
        x = torch.reshape(x, (input.shape[0],512,4,33)) #512, 4, 17 for 256 #512,4,33 for 512 hop
        for conv in self.convs:
            x = conv(x)
        x = self.final_layer(x)
        return x

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channel, 
        out_channel, 
        kernel_size, 
        stride=1, 
        padding=0, 
        z_pad=0,
        output_padding=0,
        bias=True,
        activation_function="ReLU",
        transpose=False
    ):
        super().__init__()

        self.activation_function = get_active_func(activation_function)
        self.transpose = transpose
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.bias = bias

        self.zero_pad_2d = nn.ZeroPad2d((0,0,0,0))

        if transpose == False:
            self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
                [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))
        
        if z_pad != 0:
            self.zero_pad_2d = nn.ZeroPad2d(z_pad)


        if self.transpose == True:
            self.layer_seq = nn.Sequential(
                self.zero_pad_2d,
                nn.ConvTranspose2d(
                self.in_channel,
                self.out_channel,
                kernel_size=self.kernel_size,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding), 
                self.activation_function
            )
        else:
            self.layer_seq = nn.Sequential(
                self.zero_pad_2d,
                nn.Conv2d(
                self.in_channel,
                self.out_channel,
                kernel_size=self.kernel_size,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding),
                self.activation_function
            )

    def forward(self, input):
        out = self.layer_seq(input)
        return out




