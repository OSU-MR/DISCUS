import torch
import torch.nn as nn
from .common import *

def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        output_act=1, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    
    # no. of DOWN-UP-SKIP blocks:
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    
    # No . of input channels (channels of z)
    input_depth = num_input_channels # init for 1st block
    
    # for loop on no. of DOWN-UP-SKIP blocks:
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential() # without skip block
        skip = nn.Sequential() # skip block

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper)) # include skip (concat from common.py)
        else:
            model_tmp.add(deeper)
        
        
        # bn layer of US part:
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        
        
        
        
        ## skip part:
        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
        ##
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        
        
       
    
    
        ## ds part
        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i])) # both convo + DS
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        ##
        
        

        deeper_main = nn.Sequential()

        ## last block
        if i == len(num_channels_down) - 1:
            # The deepest 
            
            
            
            
            k = num_channels_down[i]
            ## k = num_channels_down[i] + Nz # img. spec. code vector channels 
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

          
        
        
        ## US part    
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        
        # after the batch norm layer defined above
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

            
            
        input_depth = num_channels_down[i] # for next iteration of for loop
        model_tmp = deeper_main
        ##
        
        
        
        
        # for loop ends here
        
        

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)) # 
    
    
    if output_act==1:
        model.add(nn.Sigmoid())
    elif output_act==2: # ra: added this
        model.add(nn.Tanh())
    return model



