import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
#from PIL import Image
#import PIL
import pywt
import time
from scipy import signal


import matplotlib.pyplot as plt

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_params_AdaIN(opt_over, net, mlp, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif opt == 'mlp':
            params += [x for x in mlp.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            for inputs in net_input:
                inputs.requires_grad = True
                params += [inputs]
        else:
            assert False, 'what is it?'
    
    # for opt in opt_over_list:
    #     if opt == 'net':
    #         params += [x for x in net.parameters() ]
    #     elif  opt=='down':
    #         assert downsampler is not None
    #         params = [x for x in downsampler.parameters()]
    #     elif opt == 'input':
    #         net_input.requires_grad = True
    #         params += [net_input]
    #     else:
    #         assert False, 'what is it?'
            
    return params

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            for inputs in net_input:
                inputs.requires_grad = True
                params += [inputs]
        else:
            assert False, 'what is it?'
    
    # for opt in opt_over_list:
    #     if opt == 'net':
    #         params += [x for x in net.parameters() ]
    #     elif  opt=='down':
    #         assert downsampler is not None
    #         params = [x for x in downsampler.parameters()]
    #     elif opt == 'input':
    #         net_input.requires_grad = True
    #         params += [net_input]
    #     else:
    #         assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 2) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    elif method == 'hybrid': 
        assert input_depth+1 == 16
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        
        Xs = np.sin(1*2*np.pi*X)
        Ys = np.sin(1*2*np.pi*Y)
        Xc = np.cos(1*2*np.pi*X)
        Yc = np.cos(1*2*np.pi*Y)
        meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        net_input_1=  np_to_torch(meshgrid)

        Xs = np.sin(2*2*np.pi*X)
        Ys = np.sin(2*2*np.pi*Y)
        Xc = np.cos(2*2*np.pi*X)
        Yc = np.cos(2*2*np.pi*Y)
        meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        net_input_2=  np_to_torch(meshgrid)

        Xs = np.sin(4*2*np.pi*X)
        Ys = np.sin(4*2*np.pi*Y)
        Xc = np.cos(4*2*np.pi*X)
        Yc = np.cos(4*2*np.pi*Y)
        meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        net_input_3=  np_to_torch(meshgrid)

        # Xs = np.sin(8*2*np.pi*X)
        # Ys = np.sin(8*2*np.pi*Y)
        # Xc = np.cos(8*2*np.pi*X)
        # Yc = np.cos(8*2*np.pi*Y)
        # meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        # net_input_4=  np_to_torch(meshgrid)

        # Xs = np.sin(16*2*np.pi*X)
        # Ys = np.sin(16*2*np.pi*Y)
        # Xc = np.cos(16*2*np.pi*X)
        # Yc = np.cos(16*2*np.pi*Y)
        # meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        # net_input_5=  np_to_torch(meshgrid)

        # Xs = np.sin(32*2*np.pi*X)
        # Ys = np.sin(32*2*np.pi*Y)
        # Xc = np.cos(32*2*np.pi*X)
        # Yc = np.cos(32*2*np.pi*Y)
        # meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        # net_input_6=  np_to_torch(meshgrid)

        # Xs = np.sin(64*2*np.pi*X)
        # Ys = np.sin(64*2*np.pi*Y)
        # Xc = np.cos(64*2*np.pi*X)
        # Yc = np.cos(64*2*np.pi*Y)
        # meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        # net_input_7=  np_to_torch(meshgrid)

        # Xs = np.sin(128*2*np.pi*X)
        # Ys = np.sin(128*2*np.pi*Y)
        # Xc = np.cos(128*2*np.pi*X)
        # Yc = np.cos(128*2*np.pi*Y)
        # meshgrid = np.concatenate([Xs[None,:], Ys[None,:], Xc[None,:], Yc[None,:]])
        # net_input_8=  np_to_torch(meshgrid)

        shape = [1, 3, spatial_size[0], spatial_size[1]]
        net_input_9 = torch.zeros(shape)
        fill_noise(net_input_9, noise_type)
        net_input_9 *= var

        net_input = torch.cat((net_input_1, net_input_2, net_input_3, net_input_9), 1)
        # net_input = torch.cat((net_input_1, net_input_2, net_input_3, net_input_4, net_input_5, net_input_6, net_input_7, net_input_8), 1)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter, WtD):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR, weight_decay=WtD)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)#<--ra: added this
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            
            optimizer.step()
            scheduler.step() #<--ra: added this
    else:
        assert False
        
        

        
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
#             self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
#             self.save_checkpoint(val_loss, model)
            self.counter = 0
    
        self.best_score = score

        

def optimize_earlystopping(optimizer_type, parameters, closure, LR, num_iter, WtD, patience, delta, data_path, net, opt, N, NInd):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR, weight_decay=WtD)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)#<--ra: added this
        for j in range(num_iter):
            optimizer.zero_grad()
            loss = closure()
            early_stopping(loss)
            if early_stopping.early_stop:
                print("Early stopping at iteration ", j, "Model saved")
                #break
                torch.save(net, data_path + 'model_'+'opt_%d_N_%d_Ind_%d'+'EarlyStopped' % (opt, N, NInd))


            
            optimizer.step()
            scheduler.step() #<--ra: added this
    else:
        assert False


# Added these helper function (RA)
def takeMag(x): # Returns magnitude images when real/imag and interleaved even/odd along axis=0
  xMag = np.zeros((int(x.shape[0]/2), x.shape[1], x.shape[2]))
  for i in range(xMag.shape[0]):
    xMag[i,:,:] = np.sqrt(x[2*i,:,:]**2 + x[2*i+1,:,:]**2)
  return xMag

def funA(x,msk): # forward operator
  y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x),norm='ortho')) # torch_to_np(fft2c_ra(np_to_torch(x[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
  y = y*msk
  return y

def funAt(y,msk): # adjoint of the forward operator
  y = y*msk
  x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y),norm='ortho')) #torch_to_np(ifft2c_ra(np_to_torch(y[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
  return x

def swt2_haar(x): # uwt
  B = 4 # number of bands
  coef = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  haar_fil=np.zeros((B,3,3)).astype(complex)
  haar_fil[0,:,:] = [[+1/2,+1/2, 0], [+1/2,+1/2,0], [0,0,0]]
  haar_fil[1,:,:] = [[+1/2,+1/2, 0], [-1/2,-1/2,0], [0,0,0]]
  haar_fil[2,:,:] = [[+1/2,-1/2, 0], [+1/2,-1/2,0], [0,0,0]]
  haar_fil[3,:,:] = [[+1/2,-1/2, 0], [-1/2,+1/2,0], [0,0,0]]
  for i in range(B):
    coef[i,:,:] = signal.convolve2d(x, haar_fil[i,:,:], mode='same', boundary='wrap')
  return coef

def iswt2_haar(coef): # inverse uwt
  B = 4
  x = np.zeros((coef.shape[1],coef.shape[2])).astype(complex)
  haar_fil=np.zeros((B,3,3)).astype(complex)
  haar_fil[0,:,:] = np.flipud(np.fliplr([[+1/2,+1/2, 0], [+1/2,+1/2,0], [0,0,0]]))
  haar_fil[1,:,:] = np.flipud(np.fliplr([[+1/2,+1/2, 0], [-1/2,-1/2,0], [0,0,0]]))
  haar_fil[2,:,:] = np.flipud(np.fliplr([[+1/2,-1/2, 0], [+1/2,-1/2,0], [0,0,0]]))
  haar_fil[3,:,:] = np.flipud(np.fliplr([[+1/2,-1/2, 0], [-1/2,+1/2,0], [0,0,0]]))
  for i in range(B):
    x = x + signal.convolve2d(coef[i,:,:], haar_fil[i,:,:], mode='same', boundary='wrap')
  x = x/B
  return x

def st(coef, tau): # soft-thresholding
  B = 4
  for i in range(B):
    if i==0:
      coef[i,:,:] = np.maximum(np.abs(coef[i,:,:]) - tau*1e-1, 0) * np.exp(1j*np.angle(coef[i,:,:]))
    else:
      coef[i,:,:] = np.maximum(np.abs(coef[i,:,:]) - tau, 0) * np.exp(1j*np.angle(coef[i,:,:]))
  return coef

def admm_l1(x, y, msk, nIter, ss, mu, tau):
  B = 4
  u = funAt(y, msk)
  d = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  b = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  for i in range(nIter[0]):
    if i==0 or (i+1) % 10 ==0:
      print('Iteration: %2d ' %(i+1))
    for j in range(nIter[1]): #ra: hardcoded?
      gradA = funAt(funA(u, msk) - y, msk)
      gradW = mu * iswt2_haar(swt2_haar(u) - d + b)
      u = u - ss * (gradA + gradW)
    d = st(swt2_haar(u) + b, tau/mu)
    b = b + (swt2_haar(u) - d)
  return u