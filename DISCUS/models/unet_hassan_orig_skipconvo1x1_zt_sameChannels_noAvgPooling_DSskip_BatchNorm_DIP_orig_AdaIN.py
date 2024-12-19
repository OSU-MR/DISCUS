"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =True
# dtype = torch.cuda.FloatTensor


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int, # not including sparse channels
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        bias: bool = True, 
        pad: str = 'reflection', 
        act_fun: str = 'LeakyReLU',
        upsample_mode: str = 'nearest', 
        drop_prob: float = 0.0,
        bottleneck_chans: int = 1, # z image-specific
        instance_normalization: int = 1,
        mult_only: int = 1
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.bias = bias
        self.pad = pad
        self.act_fun = act_fun
        self.upsample_mode = upsample_mode
        self.drop_prob = drop_prob
        self.bottleneck_chans = bottleneck_chans
        self.instance_normalization = instance_normalization
        self.mult_only = mult_only
        
#         self.bn_layer = nn.Sequential(nn.BatchNorm2d(ch_IN*2))

        ch = chans
        if instance_normalization==1:
            ch_IN = 2*ch
        else:
            ch_IN=ch
            
        self.conv_skip = nn.ModuleList([Single_ConvBlock(in_chans, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob, instance_normalization=instance_normalization)])
        self.conv_ds = nn.ModuleList([Single_ConvBlock(in_chans, ch, kernal_size=3, stride=2, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob, instance_normalization=instance_normalization)])
        self.conv = nn.ModuleList([Single_ConvBlock(ch_IN, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob, instance_normalization=instance_normalization)])
        self.conv_up = nn.ModuleList([Single_ConvBlock_BN(ch_IN*2 + self.bottleneck_chans, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob)])  
        self.conv_skip_up = nn.ModuleList([Single_ConvBlock_BN(ch, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob)])
        self.bn_layer_bottleneck = nn.ModuleList([BN_BottleNeck(ch_IN*2 + self.bottleneck_chans)])
        self.bn_layer = nn.ModuleList([BN(ch_IN*2)])

       
        ## skip convo: doesnt change spatial or channel dimensions of output of DS layer

        for _ in range(num_pool_layers - 1):
            self.conv_skip.append(Single_ConvBlock(ch_IN, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob, instance_normalization=instance_normalization))
            self.conv_ds.append(Single_ConvBlock(ch_IN, ch, kernal_size=3, stride=2, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob, instance_normalization=instance_normalization))
            self.conv.append(Single_ConvBlock(ch_IN, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob, instance_normalization=instance_normalization))
            self.conv_up.append(Single_ConvBlock_BN(ch_IN+ch, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob))  
            self.conv_skip_up.append(Single_ConvBlock_BN(ch, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob))
            self.bn_layer_bottleneck.append(BN_BottleNeck(ch_IN + ch))
            self.bn_layer.append(BN_BottleNeck(ch_IN + ch))


            
            
#         self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        
        
        
#         self.conv_skip = nn.ModuleList()
#         for i in range(num_pool_layers - 1):
#             self.conv_skip.append(ConvBlock(ch * 2, ch, drop_prob))
#             ch //= 2


            
            
            

#         self.up_conv = nn.ModuleList()
#         self.up_transpose_conv = nn.ModuleList()
#         for i in range(num_pool_layers - 1):
#             if i==0:
#                 inp_ch = 2*ch  + bottleneck_chans
#             else:
#                 inp_ch = 2*ch 
                    
#             self.up_transpose_conv.append(TransposeConvBlock(inp_ch, ch))
#             self.up_conv.append(ConvBlock(ch, ch, drop_prob))

            
#         self.up_transpose_conv.append(TransposeConvBlock(2*ch , ch))
#         self.up_conv.append(
#             nn.Sequential(
#                 ConvBlock(ch , ch, drop_prob),
#                 nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
#             )
#         )
        
#         self.conv_bottleneck = ConvBlock_Bottleneck(1, 1, drop_prob,num_pool_layers)

#         self.bn_layer_bottleneck = nn.Sequential(nn.BatchNorm2d(ch*2 + self.bottleneck_chans))
        self.convo_layer = nn.Sequential(nn.Conv2d(ch, out_chans, kernel_size=1, stride=1, padding=0, bias=self.bias))
        self.upsample_layer = nn.Sequential(nn.Upsample(scale_factor=2, mode=self.upsample_mode))


        
    def forward(self, image: torch.Tensor, sparse_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
#         output = image[:,0:self.in_chans-1,:,:]
        output = image
    
#         print(sparse_z.size())

        
#         print("Input Image to 1st DS Block", output.size())

            # sparse_z: gating signal coming from MLP_AdaIN
            # sparse_z: (1, 128)
        if self.mult_only:
            zt_resized = torch.unsqueeze(torch.unsqueeze(torch.reshape(torch.squeeze(sparse_z, dim=0), (1, self.chans)), dim=2), dim=3)
        else:
            zt_resized = torch.unsqueeze(torch.unsqueeze(torch.reshape(torch.squeeze(sparse_z, dim=0), (2, self.chans)), dim=2), dim=3)


        # apply down-sampling layers
        for skip_layer, ds_layer, conv_layer in zip(self.conv_skip, self.conv_ds, self.conv):


            skip_output = skip_layer(output)
            # print(skip_output.size())


            stack.append(skip_output) # to concat
#             print(output.size())

            output = ds_layer(output)
            # print(output.size())
            
               
#             print(output.size())

            
            output = conv_layer(output)
            # print(output.size())

            # normalize :
            ins_mean = torch.squeeze(torch.mean(output, dim=(-1,-2)))
            ins_std = torch.squeeze(torch.std(output, dim=(-1,-2)))
#             print(ins_std.size())
            ins_mean_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_mean, dim=0), dim=2), dim=3)
            ins_std_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_std, dim=0), dim=2), dim=3)
#             print(zt_resized.size())
            if self.mult_only:
                output = output / (ins_std_resized + 1e-10)
                output = output * zt_resized[0,:,:,:] # channel modulation
            else:
                output = (output - ins_mean_resized) / (ins_std_resized + 1e-10)
                output = (output * zt_resized[0,:,:,:]) + zt_resized[1,:,:,:] # channel modulation
                       


#         output = self.conv(output)
        
        ## here at bottleneck... i can concatenate z_t with output before passing to upsampling layer...

#         print output size here
#         print("Input to US block", output.size())
        
        ## concat channels to input to upsample trans convo
        
        
#         self.layers = nn.Sequential(
#         nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
#         nn.InstanceNorm2d(out_chans),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Dropout2d(drop_prob))
        
        
        
        
        
#         z_pool = torch.unsqueeze(F.avg_pool2d(image[:,-1,:,:], kernel_size=2**(self.num_pool_layers), stride=2**(self.num_pool_layers), padding=0), axis=1) # DS happens here: self.num_pool_layers
# #         print("z pool", z_pool.size())  


#         z = torch.zeros(1,Nz*N,n[0],n[1]).type(dtype) # all zeros
#         for i in range(N):
#           z[:,i*Nz:(i+1)*Nz,:,:] = 1*(0.8*z0 + 0.2*(get_noise(Nz, INPUT, x.shape[1:], var=1./10).type(dtype) - 1/20))



#         z_pool = self.conv_bottleneck(torch.unsqueeze(image[:,-1,:,:], axis=1))
        if self.bottleneck_chans != 0:
#             if MLP==1:
#                 z_pool = MLP_bottleneck(sparse_z)
#             else:       
            z_pool = sparse_z
                
            output = torch.cat([z_pool, output], dim=1)
        
        

        
        
        
        # apply up-sampling layers
        for bn_bottleneck,bn, conv_up_layer, skip_layer_up in zip(self.bn_layer_bottleneck, self.bn_layer, self.conv_up, self.conv_skip_up):
#             print('1')
            to_cat = stack.pop()
#             print(output.size())
#             print(to_cat.size())
            output = self.upsample_layer(output)

            # call the concat function:
            output = Concat([output, to_cat], dim=1)
            # print(output.size())

            # output = torch.cat([output, to_cat], dim=1)
            
            if self.bottleneck_chans != 0:
                output = bn_bottleneck(output)
#                 self.bottleneck_chans = 0
            else:
                output = bn(output) 

#             print(output.size())
            output = conv_up_layer(output)
    
    
            
               

            output = skip_layer_up(output)
            
            ins_mean = torch.squeeze(torch.mean(output, dim=(-1,-2)))
            ins_std = torch.squeeze(torch.std(output, dim=(-1,-2)))
#             print(ins_std.size())
            ins_mean_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_mean, dim=0), dim=2), dim=3)
            ins_std_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_std, dim=0), dim=2), dim=3)

            if self.mult_only:
                output = output / (ins_std_resized + 1e-10)
                output = output * zt_resized[0,:,:,:] # channel modulation
            else:
                output = (output - ins_mean_resized) / (ins_std_resized + 1e-10)
                output = (output * zt_resized[0,:,:,:]) + zt_resized[1,:,:,:] # channel modulation
                
            
            
            ## add convo layers to downsample output before passing to US layer
#             skip_convo = downsample_layer

            
#             print(output.size())
#             print(downsample_layer.size())
            
            
            # reflect pad on the right/botton if needed to handle odd input dimensions
#             padding = [0, 0, 0, 0]
#             if output.shape[-1] != downsample_layer.shape[-1]:
#                 padding[1] = 1  # padding right
#             if output.shape[-2] != downsample_layer.shape[-2]:
#                 padding[3] = 1  # padding bottom
#             if torch.sum(torch.tensor(padding)) != 0:
#                 output = F.pad(output, padding, "reflect")

                
#             print(torch.sum(torch.tensor(padding)))
#             print(output.size())
#             print(downsample_layer.size())
            
            
#             output = torch.cat([output, downsample_layer], dim=1)
#             print("concat DS and US output for next US block",output.size())

#             output = transpose_conv(output)


#             print("output image",output.size())

        
#         output = F.ReflectionPad2d(output, 0)
        output = self.convo_layer(output)

        return output





def Concat(inputs, dim):
    # def __init__(self, dim, *args):
    #     super(Concat, self).__init__()
    #     self.dim = dim

    #     for idx, module in enumerate(args):
    #         self.add_module(str(idx), module)

    # def forward(self, input):
        # inputs = []
        # for module in self._modules.values():
        #     inputs.append(module(input))

    inputs_shapes2 = [x.shape[2] for x in inputs]
    inputs_shapes3 = [x.shape[3] for x in inputs]        

    if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
        inputs_ = inputs
    else:
        target_shape2 = min(inputs_shapes2)
        target_shape3 = min(inputs_shapes3)

        inputs_ = []
        for inp in inputs: 
            diff2 = (inp.size(2) - target_shape2) // 2 
            diff3 = (inp.size(3) - target_shape3) // 2 
            inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

    return torch.cat(inputs_, dim=dim)

    # def __len__(self):
    #     return len(self._modules)



    
class MLP_AdaIN(nn.Module):
    """
    A Single Convolutional Block that consists of one ReflectionPad2d layer followed by convolution,
    batch normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, chans: int, z_size: tuple, mult_only: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.chans = chans
        self.nz = z_size
        
        self.inter_chans = np.power(2, [10, 9, 8])

        self.mult_only = mult_only
        if self.mult_only:
            self.output_ch = self.chans
        else:
            self.output_ch = 2*self.chans



       
            
        self.layers = nn.Sequential(
            nn.Flatten(2, -1),
            nn.Linear(self.nz[0]*self.nz[1], self.inter_chans[0]),
            nn.ReLU(),
            nn.Linear(self.inter_chans[0], self.inter_chans[1]),
            nn.ReLU(),
            nn.Linear(self.inter_chans[1], self.inter_chans[2]),
            nn.ReLU(),
            nn.Linear(self.inter_chans[2], self.output_ch),
            nn.ReLU()
        )

        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
#         print(image.size())
        output = self.layers(image)
#         print(output.size())

        return output
#         print(output.size())
#         return torch.reshape(output, (-1, self.siz0, self.siz1))

    
    


    
class MLP_BottleNeck(nn.Module):
    """
    A Single Convolutional Block that consists of one ReflectionPad2d layer followed by convolution,
    batch normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, num_pool_layers: int,bottleneck_chans:int, img_size: tuple, z_size: tuple):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.NLy = num_pool_layers
        self.b_chans = bottleneck_chans
        self.n = img_size
        self.nz = z_size
        self.siz0 = int(self.n[0]/2**(self.NLy))
        self.siz1 = int(self.n[1]/2**(self.NLy))
        self.siz_inter = 128

       
            
        self.layers = nn.Sequential(
            nn.Flatten(2, -1),
            nn.Linear(self.nz[0]*self.nz[1], self.siz_inter),
            nn.BatchNorm1d(self.b_chans),
            nn.ReLU(),
            nn.Linear(self.siz_inter, self.siz0*self.siz1),
            nn.BatchNorm1d(self.b_chans),
            nn.ReLU()
            
        )

        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
#         print(image.size())
        output = self.layers(image)
#         print(output.size())
        return torch.reshape(output, (-1, self.siz0, self.siz1))

    
 
    
class BN_BottleNeck(nn.Module):
    """
    A Single Convolutional Block that consists of one ReflectionPad2d layer followed by convolution,
    batch normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.out_chans = out_chans
       
            
        self.layers = nn.Sequential(
            nn.BatchNorm2d(out_chans)
        )

        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)

  


class BN(nn.Module):
    """
    A Single Convolutional Block that consists of one ReflectionPad2d layer followed by convolution,
    batch normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.out_chans = out_chans
       
            
        self.layers = nn.Sequential(
            nn.BatchNorm2d(out_chans)
        )

        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)

    
    


class Single_ConvBlock(nn.Module):
    """
    A Single Convolutional Block that consists of one ReflectionPad2d layer followed by convolution,
    batch normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, kernal_size: int, 
                stride: int, pad: str, bias: bool, act_fun: str, drop_prob: float, instance_normalization: int=0):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = pad
        self.bias = bias
        self.act_fun = act_fun
        self.drop_prob = drop_prob
        self.instance_normalization = instance_normalization
        
        if self.padding=='reflection':
            to_pad = int((kernal_size - 1) / 2)
            
        
            
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(to_pad),
            nn.Conv2d(in_chans, out_chans, kernal_size, stride, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True)) 
        
        self.IN = nn.InstanceNorm2d(out_chans, affine=True)
        self.BN = nn.BatchNorm2d(out_chans)
        self.DO = nn.Dropout2d(drop_prob)


        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        out = self.layers(image)
        
        if self.instance_normalization==1:
            out1 = self.BN(out)
            out2 = self.IN(out)
            output = torch.cat((out1, out2),dim=1)
        else:
            output = self.BN(out)
        output = self.DO(output)
        
        return output


    
class Single_ConvBlock_BN(nn.Module):
    """
    A Single Convolutional Block that consists of one ReflectionPad2d layer followed by convolution,
    batch normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, kernal_size: int, 
                stride: int, pad: str, bias: bool, act_fun: str, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = pad
        self.bias = bias
        self.act_fun = act_fun
        self.drop_prob = drop_prob
        
        if self.padding=='reflection':
            to_pad = int((kernal_size - 1) / 2)
            
        
            
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(to_pad),
            nn.Conv2d(in_chans, out_chans, kernal_size, stride, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_chans),
            nn.Dropout2d(drop_prob)
        ) 
        
#         self.IN = nn.InstanceNorm2d(out_chans, affine=True)
#         self.BN = nn.BatchNorm2d(out_chans)
#         self.DO = nn.Dropout2d(drop_prob)


        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        out = self.layers(image)
        
#         if self.instance_normalization==1:
#             out1 = self.BN(out)
#             out2 = self.IN(out)
#             output = torch.cat((out1, out2),dim=1)
#         else:
#             output = self.BN(out)
#         output = self.DO(output)
        
        return out


    
    
    
    
# class Upsample_ConvBlock(nn.Module):
#     """
#     A Single Convolutional Block that consists of one upsample layer followed by concat, and
#     batch normalization.
#     """

#     def __init__(self, in_chans: int, out_chans: int, kernal_size: int, 
#                 stride: int, pad: str, bias: bool, act_fun: str, drop_prob: float):
#         """
#         Args:
#             in_chans: Number of channels in the input.
#             out_chans: Number of channels in the output.
#             drop_prob: Dropout probability.
#         """
#         super().__init__()

#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.kernal_size = kernal_size
#         self.stride = stride
#         self.padding = pad
#         self.bias = bias
#         self.act_fun = act_fun
#         self.drop_prob = drop_prob
        
#         if self.padding=='reflection':
#             to_pad = int((kernal_size - 1) / 2)
            
#         self.layers = nn.Sequential(
#             nn.ReflectionPad2d(to_pad),
#             nn.Conv2d(in_chans, out_chans, kernal_size, stride, padding=0, bias=bias),
#             nn.LeakyReLU(0.2, inplace=True),            
#             nn.BatchNorm2d(out_chans),
#             nn.Dropout2d(drop_prob),
#         )

        
#     def forward(self, image: torch.Tensor, to_cat: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: Input 4D tensor of shape `(N, in_chans, H, W)`.
#         Returns:
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#         return self.layers(image)







class ConvBlock_general(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, num_net_layers: int, kernel_size: int, padding: int, stride: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.num_net_layers = num_net_layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, stride = stride, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size, padding=padding, stride = stride, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)



    
class ConvBlock_DS(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)

    
    

    
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


    
    
    
    
class ConvBlock_1x1(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)

    
    
class ConvBlock_Bottleneck(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, num_pool_layers: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.num_pool_layers = num_pool_layers

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=2**(self.num_pool_layers), stride=2**(self.num_pool_layers), padding=0, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


    
    
class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
    
    
    
def act(act_fun: str):
    if act_fun == 'LeakyReLU':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_fun == 'Swish':
        return Swish()
    elif act_fun == 'ELU':
        return nn.ELU()
    
# def upsample(factor: int = 2, mode: str = 'nearest'):
#     return nn.Upsample(scale_factor=factor, mode=mode)

