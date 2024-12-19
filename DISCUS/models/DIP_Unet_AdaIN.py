"""
Base DIP Network Architecture as used in "Ulyanov, D.; Vedaldi, A.; Lempitsky, V. Deep Image Prior.
 arXiv 2017, arXiv: 1711.10925v3" with Adaptive Instance Normalization (AdaIN) or Channel Modulation as described in
 "Style GAN. arXiv:1812.04948v3 [cs.NE] 29 Mar 2019"
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =True
# dtype = torch.cuda.FloatTensor



# accepts Odd image sizes
# input image size should not be multiple of 2 or 32 or 64. It can be any
# Only condition to satisfy is spatial size >= 2^(NLy) since Down Sampling NLy times by a factor of 2 
class Unet(nn.Module):
    """
    Unet Model: Base DIP Network Architecture as used in "Ulyanov, D.; Vedaldi, A.; Lempitsky, V. Deep Image Prior.
 arXiv 2017, arXiv: 1711.10925v3" with AdaIN or channel modulationb as described in
 "Style GAN. arXiv:1812.04948v3 [cs.NE] 29 Mar 2019"

    """

    def __init__(
        self,
        in_chans: int = 3, # Num of input channels; excluding sparse channels for MLP
        out_chans: int = 2, # Num of output channels; 2/1 for complex/real images
        chans: int = 128, # Num of output channels of 1st block; constant throughout network
        num_pool_layers: int = 6, # num of down-sampling and up-sampling layers 
        bias: bool = True, # Bias for Conv2d layer
        pad: str = 'reflection', # Pading for Conv2d layer 
        act_fun: str = 'LeakyReLU', # Non-linear Activation layer
        upsample_mode: str = 'nearest', # Upsampling layer mode
        drop_prob: float = 0.0, # dropout probability e.g. 0.01 means 1 %
        mult_only: int = 1 # wheteher to apply both multiplicative and aditive modulaion or mult. only
    ):

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
        self.mult_only = mult_only
        

        ch = chans # Num of channels; same throghout network

        ## DS modules: 1st layers... input channels   
        self.conv_skip = nn.ModuleList([Single_ConvBlock(in_chans, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob)])
        self.conv_ds = nn.ModuleList([Single_ConvBlock(in_chans, ch, kernal_size=3, stride=2, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob)])
        self.conv = nn.ModuleList([Single_ConvBlock(ch, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob)])
        
        ## US modules: 1st layers... 
        self.conv_up = nn.ModuleList([Single_ConvBlock(ch*2, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob)])  
        self.conv_skip_up = nn.ModuleList([Single_ConvBlock(ch, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob)])
        self.bn_layer = nn.ModuleList([BN(ch*2)])

       

        for _ in range(num_pool_layers - 1):
            ## DS modules: add remaining layers...    
            self.conv_skip.append(Single_ConvBlock(ch, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob))
            self.conv_ds.append(Single_ConvBlock(ch, ch, kernal_size=3, stride=2, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob))
            self.conv.append(Single_ConvBlock(ch, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob))
            
            ## US modules: add reamining layers...
            self.conv_up.append(Single_ConvBlock(2*ch, ch, kernal_size=3, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob))  
            self.conv_skip_up.append(Single_ConvBlock(ch, ch, kernal_size=1, stride=1, pad=pad, bias=bias, act_fun = act_fun, drop_prob = drop_prob))
            self.bn_layer.append(BN(2*ch))

            
        ## Last Convo layer:
        self.convo_layer = nn.Sequential(nn.Conv2d(ch, out_chans, kernel_size=1, stride=1, padding=0, bias=self.bias))

        ## US layer
        self.upsample_layer = nn.Sequential(nn.Upsample(scale_factor=2, mode=self.upsample_mode))




        
    def forward(self, image: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        stack = [] # to concat with US 
        output = image

        # z_MLP: gating signal coming from MLP_AdaIN
        # z_MLP: (1, chans) # assuming batch size=1
        z_MLP = z
        if self.mult_only:
            zt_resized = torch.unsqueeze(torch.unsqueeze(torch.reshape(torch.squeeze(z_MLP, dim=0), (1, self.chans)), dim=2), dim=3)
        else: # both mult and add
            zt_resized = torch.unsqueeze(torch.unsqueeze(torch.reshape(torch.squeeze(z_MLP, dim=0), (2, self.chans)), dim=2), dim=3)


        ## apply down-sampling layers
        for skip_layer, ds_layer, conv_layer in zip(self.conv_skip, self.conv_ds, self.conv):
            skip_output = skip_layer(output)
            stack.append(skip_output) # to concat with US

            output = ds_layer(output)
            output = conv_layer(output)

            ## Output Modulation with AdaIN Gating signal from MLP...
            # normalize :
            ins_mean = torch.squeeze(torch.mean(output, dim=(-1,-2)))
            ins_std = torch.squeeze(torch.std(output, dim=(-1,-2)))
            # reshape across channels
            ins_mean_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_mean, dim=0), dim=2), dim=3)
            ins_std_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_std, dim=0), dim=2), dim=3)
            if self.mult_only:
                output = output / (ins_std_resized + 1e-10)
                output = output * zt_resized[0,:,:,:] # channel modulation
            else: # both mult and add
                output = (output - ins_mean_resized) / (ins_std_resized + 1e-10)
                output = (output * zt_resized[0,:,:,:]) + zt_resized[1,:,:,:] # channel modulation
                       


        
        ## apply up-sampling layers
        for bn_layer, conv_up_layer, skip_layer_up in zip(self.bn_layer, self.conv_up, self.conv_skip_up):
            to_cat = stack.pop() # stored from DS layer
            output = self.upsample_layer(output)

            # call the concat function:
            output = Concat([output, to_cat], dim=1) # Robust to Odd image sizes
            # output = torch.cat([output, to_cat], dim=1)

            output = bn_layer(output) 
            output = conv_up_layer(output)
            output = skip_layer_up(output)
            

            ## Output Modulation with AdaIN Gating signal...
            # normalize :
            ins_mean = torch.squeeze(torch.mean(output, dim=(-1,-2)))
            ins_std = torch.squeeze(torch.std(output, dim=(-1,-2)))
            # reshape across channels
            ins_mean_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_mean, dim=0), dim=2), dim=3)
            ins_std_resized = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(ins_std, dim=0), dim=2), dim=3)
            if self.mult_only:
                output = output / (ins_std_resized + 1e-10)
                output = output * zt_resized[0,:,:,:] # channel modulation
            else: # both mult and add
                output = (output - ins_mean_resized) / (ins_std_resized + 1e-10)
                output = (output * zt_resized[0,:,:,:]) + zt_resized[1,:,:,:] # channel modulation
                
            
 
        output = self.convo_layer(output)

        return output







class MLP_AdaIN(nn.Module):
    """
    A FC Block that consists of multiple units having a Linear layer and a non-linear activation.
    Getting a gating signal for Adaptive Instance Normalization (AdaIN) or Channel Modulation as described in
 "Style GAN. arXiv:1812.04948v3 [cs.NE] 29 Mar 2019" to be used with DIP as presented in "Ulyanov, D.; Vedaldi, A.; Lempitsky, V. Deep Image Prior.
 arXiv 2017, arXiv: 1711.10925v3"
    """

    def __init__(self, chans: int = 128, z_size: tuple = (32, 32), mult_only: int = 1):
        """
        Args:
            chans: Number of channels in the output.
            z_size: Spatial dimensions of input z.
            mult_only: int = 1 # wheteher to apply both multiplicative and aditive modulaion or mult. only
        """
        super().__init__()

        nz = z_size
        inter_chans = np.power(2, [12, 10, 8]) # define size of FC layers (number of neurons)

        if mult_only:
            output_ch = chans
        else:
            output_ch = 2*chans

            
        self.layers = nn.Sequential(
            nn.Flatten(2, -1),
            nn.Linear(nz[0]*nz[1], inter_chans[0]),
            nn.ReLU(),
            nn.Linear(inter_chans[0], inter_chans[1]),
            nn.ReLU(),
            nn.Linear(inter_chans[1], inter_chans[2]),
            nn.ReLU(),
            nn.Linear(inter_chans[2], output_ch),
            nn.ReLU()
        )

        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans)`.
        """

        output = self.layers(image)
        return output

    

class Single_ConvBlock(nn.Module):
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
            kernal_size, stride, pad, bias: Conv2d parameters
            act_fun: non-linear activation
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
        
        if pad=='reflection':
            to_pad = int((kernal_size - 1) / 2)
            
        
            
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(to_pad),
            nn.Conv2d(in_chans, out_chans, kernal_size, stride, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_chans),
            nn.Dropout2d(drop_prob)
            )



        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        output = self.layers(image)
        return output




class BN(nn.Module):
    """
    A Single 2d Batch Normalization layer.
    """

    def __init__(self, out_chans: int):
        """
        Args:
            out_chans: Number of channels in the output.
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

    


def Concat(inputs, dim): # concats mismatched spatial sizes 
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
    
    
    
    
