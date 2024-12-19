import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from .common_utils import *
from utils.fftc import *
import torch
import scipy.io
import sigpy as sp
from sigpy.mri import app

def create_rotations(data_path, N, Nm, angRange, sud, seed):
  # rng = np.random.default_rng(100*seed)
  # xci = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  # xci = xci - 1j*xci

  # ni = np.array(xci.shape) # original size
  # ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  # if sud==-1:
  #   n = (np.floor(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round down to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
  #   xc = xci[ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  # elif sud==1:
  #   n = (np.ceil(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round up to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
  #   xc[ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xci
  
  # xc = xc/np.max(np.abs(xc))

  # # complex to real
  # x = np.zeros((2*N, n[0], n[1]))
  # x[0,:,:] = np.real(xc)
  # x[1,:,:] = np.imag(xc)

  rng = np.random.default_rng(100*seed)
  xi = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  xi = np.tile(xi,(2,1,1)) # replicate from real to real+imaginatry
  xi[1,:,:] = xi[1,:,:]*rng.uniform(-1,1) # scale the imaginary channel

  ni = np.array((xi.shape[1], xi.shape[2])) # original size
  m = np.array(xi.shape[0])
  ctri = (np.floor(np.float32(ni)/2)).astype(int)  # image center (don't add one because indicies start from '0')
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    # xc = np.zeros((m, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    xc = xi[:, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    xc = np.zeros((m, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2)).astype(int)  # image center
    xc[:, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xi

  xmax = np.max(np.sqrt(xc[0::2,:,:]**2 + xc[1::2,:,:]**2))
  xc = xc/xmax

  # Generate additional repetions of the image
  # print(xc.shape)
  x = np.zeros((2*N, n[0], n[1]))
  x[0:2,:,:] = xc
  angle = np.zeros([N,1])
  for i in range(N-1):
    angle[i+1] = rng.uniform(-angRange/2,angRange/2)
    x[2*i+2:2*i+4,:,:] = scipy.ndimage.rotate(xc[0:2,:,:], angle[i+1].item(), axes=(2, 1),reshape=False)
  
  print('Angles: ',', '.join('%1.2f' % (angle[j]) for j in range(len(angle)))) 

  # print(x.shape)
  xdiff = x - np.tile(x[0:2,:,:],(N, 1, 1))
  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(x,[1,0,2]), [n[0],n[1]*2*N]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(xdiff,[1,0,2]), [n[0],n[1]*2*N]), vmin=-0.2, vmax=0.2, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  return x


def create_smooth_rotations(data_path, N, Nm, angRange, sud, seed):
  # rng = np.random.default_rng(100*seed)
  # xci = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  # xci = xci - 1j*xci

  # ni = np.array(xci.shape) # original size
  # ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  # if sud==-1:
  #   n = (np.floor(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round down to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
  #   xc = xci[ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  # elif sud==1:
  #   n = (np.ceil(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round up to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
  #   xc[ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xci
  
  # xc = xc/np.max(np.abs(xc))

  # # complex to real
  # x = np.zeros((2*N, n[0], n[1]))
  # x[0,:,:] = np.real(xc)
  # x[1,:,:] = np.imag(xc)

  rng = np.random.default_rng(100*seed)
  xi = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  xi = np.tile(xi,(2,1,1)) # replicate from real to real+imaginatry
  xi[1,:,:] = xi[1,:,:]*rng.uniform(-1,1) # scale the imaginary channel

  ni = np.array((xi.shape[1], xi.shape[2])) # original size
  m = np.array(xi.shape[0])
  ctri = (np.floor(np.float32(ni)/2)).astype(int)  # image center (don't add one because indicies start from '0')
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    # xc = np.zeros((m, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    xc = xi[:, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    xc = np.zeros((m, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2)).astype(int)  # image center
    xc[:, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xi

  xmax = np.max(np.sqrt(xc[0::2,:,:]**2 + xc[1::2,:,:]**2))
  xc = xc/xmax

  # Generate additional repetions of the image
  # print(xc.shape)
  x = np.zeros((2*N, n[0], n[1]))
  # x[0:2,:,:] = xc
  # angle = np.zeros([N,1])
  angle = np.linspace(-angRange/2,angRange/2, N)

  for i in range(N):
    # angle[i+1] = rng.uniform(-angRange/2,angRange/2)
    angle
    x[2*i:2*(i+1),:,:] = scipy.ndimage.rotate(xc[0:2,:,:], angle[i].item(), axes=(2, 1),reshape=False)
  
  print('Angles: ',', '.join('%1.2f' % (angle[j]) for j in range(len(angle)))) 

  # print(x.shape)
  xdiff = x - np.tile(x[0:2,:,:],(N, 1, 1))
  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(x,[1,0,2]), [n[0],n[1]*2*N]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(xdiff,[1,0,2]), [n[0],n[1]*2*N]), vmin=-0.2, vmax=0.2, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  return x


def create_translations(data_path, N, Nm, pixRange, sud, seed):
  # rng = np.random.default_rng(100*seed)
  # xci = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  # xci = xci - 1j*xci

  # ni = np.array(xci.shape) # original size
  # ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  # if sud==-1:
  #   n = (np.floor(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round down to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
  #   xc = xci[ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  # elif sud==1:
  #   n = (np.ceil(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round up to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
  #   xc[ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xci
  
  # xc = xc/np.max(np.abs(xc))

  # # complex to real
  # x = np.zeros((2*N, n[0], n[1]))
  # x[0,:,:] = np.real(xc)
  # x[1,:,:] = np.imag(xc)

  rng = np.random.default_rng(100*seed)
  xi = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  xi = np.tile(xi,(2,1,1)) # replicate from real to real+imaginatry
  xi[1,:,:] = xi[1,:,:]*rng.uniform(-1,1) # scale the imaginary channel

  ni = np.array((xi.shape[1], xi.shape[2])) # original size
  m = np.array(xi.shape[0])
  ctri = (np.floor(np.float32(ni)/2)).astype(int)  # image center (don't add one because indicies start from '0')
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    # xc = np.zeros((m, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    xc = xi[:, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    xc = np.zeros((m, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2)).astype(int)  # image center
    xc[:, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xi

  xmax = np.max(np.sqrt(xc[0::2,:,:]**2 + xc[1::2,:,:]**2))
  xc = xc/xmax

  # Generate additional repetions of the image
  # print(xc.shape)
  x = np.zeros((2*N, n[0], n[1]))
  x[0:2,:,:] = xc
  pix = np.zeros([N,2])
  for i in range(N-1):
    pix[i+1,0] = rng.uniform(-pixRange[0]/2,pixRange[0]/2)
    pix[i+1,1] = rng.uniform(-pixRange[1]/2,pixRange[1]/2)

    x[2*i+2:2*i+4,:,:] = scipy.ndimage.shift(xc[0:2,:,:], (0,pix[i+1,0].item(), pix[i+1,1].item()))
  
  print('Pixel shift: ',', '.join('(%1.2f, %1.2f)' % (pix[j, 0], pix[j, 1]) for j in range(len(pix[:,0])))) 

  # print(x.shape)
  xdiff = x - np.tile(x[0:2,:,:],(N, 1, 1))
  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(x,[1,0,2]), [n[0],n[1]*2*N]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(xdiff,[1,0,2]), [n[0],n[1]*2*N]), vmin=-0.2, vmax=0.2, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  return x


def create_translations_1d(data_path, N, Nm, pixRange, direction, sud, seed):
  # rng = np.random.default_rng(100*seed)
  # xci = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  # xci = xci - 1j*xci

  # ni = np.array(xci.shape) # original size
  # ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  # if sud==-1:
  #   n = (np.floor(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round down to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
  #   xc = xci[ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  # elif sud==1:
  #   n = (np.ceil(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round up to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
  #   xc[ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xci
  
  # xc = xc/np.max(np.abs(xc))

  # # complex to real
  # x = np.zeros((2*N, n[0], n[1]))
  # x[0,:,:] = np.real(xc)
  # x[1,:,:] = np.imag(xc)

  rng = np.random.default_rng(100*seed)
  xi = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  xi = np.tile(xi,(2,1,1)) # replicate from real to real+imaginatry
  xi[1,:,:] = xi[1,:,:]*rng.uniform(-1,1) # scale the imaginary channel

  ni = np.array((xi.shape[1], xi.shape[2])) # original size
  m = np.array(xi.shape[0])
  ctri = (np.floor(np.float32(ni)/2)).astype(int)  # image center (don't add one because indicies start from '0')
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    # xc = np.zeros((m, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    xc = xi[:, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    xc = np.zeros((m, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2)).astype(int)  # image center
    xc[:, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xi

  xmax = np.max(np.sqrt(xc[0::2,:,:]**2 + xc[1::2,:,:]**2))
  xc = xc/xmax

  # Generate additional repetions of the image
  # print(xc.shape)
  x = np.zeros((2*N, n[0], n[1]))
  x[0:2,:,:] = xc
  pix = np.zeros([N,1])
  for i in range(N-1):
    pix[i+1] = rng.uniform(-pixRange/2,pixRange/2)
#     pix[i+1,1] = rng.uniform(-pixRange[1]/2,pixRange[1]/2)

    if direction:
        x[2*i+2:2*i+4,:,:] = scipy.ndimage.shift(xc[0:2,:,:], (0,0,pix[i+1].item()))
    else:
        x[2*i+2:2*i+4,:,:] = scipy.ndimage.shift(xc[0:2,:,:], (0,pix[i+1].item(),0))

  
  print('Pixel shift: ',', '.join('(%1.2f)' % pix[j] for j in range(len(pix)))) 

    
    
  # print(x.shape)
  xdiff = x - np.tile(x[0:2,:,:],(N, 1, 1))
  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(x,[1,0,2]), [n[0],n[1]*2*N]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(xdiff,[1,0,2]), [n[0],n[1]*2*N]), vmin=-0.2, vmax=0.2, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  return x

def create_rotations_and_translations1d_horiz(data_path, N, Nm, angRange, pixRange, sud, seed):
  # rng = np.random.default_rng(100*seed)
  # xci = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  # xci = xci - 1j*xci

  # ni = np.array(xci.shape) # original size
  # ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  # if sud==-1:
  #   n = (np.floor(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round down to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
  #   xc = xci[ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  # elif sud==1:
  #   n = (np.ceil(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round up to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
  #   xc[ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xci
  
  # xc = xc/np.max(np.abs(xc))

  # # complex to real
  # x = np.zeros((2*N, n[0], n[1]))
  # x[0,:,:] = np.real(xc)
  # x[1,:,:] = np.imag(xc)

  rng = np.random.default_rng(100*seed)
  xi = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  xi = np.tile(xi,(2,1,1)) # replicate from real to real+imaginatry
  xi[1,:,:] = xi[1,:,:]*rng.uniform(-1,1) # scale the imaginary channel

  ni = np.array((xi.shape[1], xi.shape[2])) # original size
  m = np.array(xi.shape[0])
  ctri = (np.floor(np.float32(ni)/2)).astype(int)  # image center (don't add one because indicies start from '0')
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    # xc = np.zeros((m, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    xc = xi[:, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    xc = np.zeros((m, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2)).astype(int)  # image center
    xc[:, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xi

  xmax = np.max(np.sqrt(xc[0::2,:,:]**2 + xc[1::2,:,:]**2))
  xc = xc/xmax

  # Generate additional repetions of the image
  # print(xc.shape)
  x = np.zeros((2*N, n[0], n[1]))
  x[0:2,:,:] = xc
    
  angle = np.zeros([N,1])
  pix = np.zeros([N,1])
    
  for i in range(N-1):
    angle[i+1] = rng.uniform(-angRange/2,angRange/2)
    pix[i+1] = rng.uniform(-pixRange/2,pixRange/2)
    # pix[i+1,1] = rng.uniform(-pixRange[1]/2,pixRange[1]/2)

    xTmp = scipy.ndimage.rotate(xc[0:2,:,:], angle[i+1].item(), axes=(2, 1),reshape=False)
    x[2*i+2:2*i+4,:,:] = scipy.ndimage.shift(xTmp, (0, 0, pix[i+1].item()))
  
  print('Angle, Pixel shift horiz.: ',', '.join('(%1.2f, %1.2f)' % (angle[j], pix[j]) for j in range(len(angle)))) 
#   print('Pixel shift: ',', '.join('(%1.2f, %1.2f)' % (pix[j, 0], pix[j, 1]) for j in range(len(pix[:,0])))) 

  # print(x.shape)
  xdiff = x - np.tile(x[0:2,:,:],(N, 1, 1))
  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(x,[1,0,2]), [n[0],n[1]*2*N]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(xdiff,[1,0,2]), [n[0],n[1]*2*N]), vmin=-0.2, vmax=0.2, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  return x

def create_rotations_and_translations2d(data_path, N, Nm, angRange, pixRange, sud, seed):
  # rng = np.random.default_rng(100*seed)
  # xci = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  # xci = xci - 1j*xci

  # ni = np.array(xci.shape) # original size
  # ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  # if sud==-1:
  #   n = (np.floor(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round down to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
  #   xc = xci[ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  # elif sud==1:
  #   n = (np.ceil(np.float32(xci.shape)/Nm)*Nm).astype(int) # Round up to an even-sized array
  #   xc = np.zeros((n[0], n[1]), dtype=np.complex_)
  #   ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
  #   xc[ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xci
  
  # xc = xc/np.max(np.abs(xc))

  # # complex to real
  # x = np.zeros((2*N, n[0], n[1]))
  # x[0,:,:] = np.real(xc)
  # x[1,:,:] = np.imag(xc)

  rng = np.random.default_rng(100*seed)
  xi = plt.imread(data_path).astype('f') # 'f' for float32 and float64 for float64 and int for integer
  xi = np.tile(xi,(2,1,1)) # replicate from real to real+imaginatry
  xi[1,:,:] = xi[1,:,:]*rng.uniform(-1,1) # scale the imaginary channel

  ni = np.array((xi.shape[1], xi.shape[2])) # original size
  m = np.array(xi.shape[0])
  ctri = (np.floor(np.float32(ni)/2)).astype(int)  # image center (don't add one because indicies start from '0')
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    # xc = np.zeros((m, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    xc = xi[:, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    xc = np.zeros((m, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2)).astype(int)  # image center
    xc[:, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xi

  xmax = np.max(np.sqrt(xc[0::2,:,:]**2 + xc[1::2,:,:]**2))
  xc = xc/xmax

  # Generate additional repetions of the image
  # print(xc.shape)
  x = np.zeros((2*N, n[0], n[1]))
  x[0:2,:,:] = xc
    
  angle = np.zeros([N,1])
  pix = np.zeros([N,2])
    
  for i in range(N-1):
    angle[i+1] = rng.uniform(-angRange/2,angRange/2)
    pix[i+1,0] = rng.uniform(-pixRange[0]/2,pixRange[0]/2)
    pix[i+1,1] = rng.uniform(-pixRange[1]/2,pixRange[1]/2)

    xTmp = scipy.ndimage.rotate(xc[0:2,:,:], angle[i+1].item(), axes=(2, 1),reshape=False)
    x[2*i+2:2*i+4,:,:] = scipy.ndimage.shift(xTmp, (0,pix[i+1,0].item(), pix[i+1,1].item()))
  
  print('Angle, Pixel shift vert., Pixel shift horiz.: ',', '.join('(%1.2f, %1.2f, %1.2f)' % (angle[j], pix[j, 0], pix[j, 1]) for j in range(len(angle)))) 
#   print('Pixel shift: ',', '.join('(%1.2f, %1.2f)' % (pix[j, 0], pix[j, 1]) for j in range(len(pix[:,0])))) 

  # print(x.shape)
  xdiff = x - np.tile(x[0:2,:,:],(N, 1, 1))
  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(x,[1,0,2]), [n[0],n[1]*2*N]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(xdiff,[1,0,2]), [n[0],n[1]*2*N]), vmin=-0.2, vmax=0.2, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  return x





def read_lge(data_path, N, Nm, E, sud, seed, scale):
  rng = np.random.default_rng(100*seed)
  xtmp = np.load(data_path) # 'f' for float32 and float64 for float64 and int for integer
  xtmp = xtmp[0:N,E,:,:] #<-- 'E=0' for t1-weight, 'E=1' for phase reference

  xi = np.zeros((2*xtmp.shape[0],xtmp.shape[1], xtmp.shape[2])) # replicate from real to real+imaginatry
  xi[0::2,:,:] = np.real(xtmp)
  xi[1::2,:,:] = np.imag(xtmp)

  ni = np.array((xi.shape[1], xi.shape[2])) # original size
  m  = np.array(xi.shape[0])

  ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    print(n)
    x = np.zeros((m, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    x = xi[:, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    x = np.zeros((m, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
    x[:, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = xi

  print("Before scaling x: ", np.min(takeMag(x)), np.max(takeMag(x)))
  xmax = np.max(np.sqrt(x[0::2,:,:]**2 + x[1::2,:,:]**2))
  x = (x/xmax)*scale
  print("After scaling x: ", np.min(takeMag(x)), np.max(takeMag(x)))


  # print(x.shape)
  xdiff = x - np.tile(x[0:2,:,:],(N, 1, 1))
  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(x,[1,0,2]), [n[0],n[1]*2*N]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(xdiff,[1,0,2]), [n[0],n[1]*2*N]), vmin=-0.2, vmax=0.2, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  return x


def read_lge_pmri(data_path, N, Nm, sud, seed):
  mat = scipy.io.loadmat(data_path)
  # xN = mat['im_sense']
  # S = mat['S']
  ytmp = mat['kspace']
  ytmp = np.squeeze(ytmp)
  ytmp = np.transpose(ytmp, (3,2,0,1))
  ytmp = ytmp[0:N, :, :, :]
  yi = np.zeros((2*ytmp.shape[0],ytmp.shape[1],ytmp.shape[2], ytmp.shape[3])) # replicate from real to real+imaginatry
  yi[0::2,:,:] = np.real(ytmp)
  yi[1::2,:,:] = np.imag(ytmp)

  ni = np.array((yi.shape[2], yi.shape[3])) # original size
  c  = np.array(yi.shape[1]) # number of coils
  m  = np.array(yi.shape[0]) # number of repetitions

  ctri = (np.floor(np.float32(ni)/2 + 1)).astype(int)  # image center
  if sud==-1:
    n = (np.floor(np.float32(ni)/Nm)*Nm).astype(int) # Round down to an even-sized array
    y = np.zeros((m, c, n[0], n[1]))
    # xc = xci[ctri[0]-(n[0]/2).astype(int) : ctri[0]+(n[0]/2).astype(int), ctri[1]-(n[1]/2).astype(int) : ctri[1]+(n[1]/2).astype(int)]
    y = yi[:, :, ctri[0]-(np.floor(n[0]/2)).astype(int) : ctri[0]+(np.ceil(n[0]/2)).astype(int), ctri[1]-(np.floor(n[1]/2)).astype(int) : ctri[1]+(np.ceil(n[1]/2)).astype(int)]
  elif sud==1:
    n = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    y = np.zeros((m, c, n[0], n[1]))
    ctr = (np.floor(np.float32(n)/2 + 1)).astype(int)  # image center
    y[:, :, ctr[0]-(np.floor(ni[0]/2)).astype(int) : ctr[0]+(np.ceil(ni[0]/2)).astype(int), ctr[1]-(np.floor(ni[1]/2)).astype(int) : ctr[1]+(np.ceil(ni[1]/2)).astype(int)] = yi

  ymax = np.max(np.sqrt(y[0::2,:,:,:]**2 + y[1::2,:,:,:]**2))
  y = 10*y/ymax

  msk = np.zeros((m,n[0],n[1]))
  tmp = np.sqrt(y[0::2,:,:,:]**2 + y[1::2,:,:,:]**2)
  tmp = np.max(tmp, axis=1)
  msk = tmp>0
  msk = np.expand_dims(msk, axis=1)

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(np.abs(y)**0.25,[1,2,0,3]), [n[0]*c,n[1]*2*N]), vmin=0, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(msk,[1,2,0,3]), [n[0],n[1]*N]), vmin=0, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()
  return y, msk;


def espirit_maps(y, cw, cp):
  n = np.array((y.shape[2], y.shape[3])) # original size
  c  = np.array(y.shape[1]) # number of coils
  m  = np.array(y.shape[0]) # number of repetitions
  yc = np.zeros((m//2,c,n[0],n[1])).astype(complex)
  yc = y[0::2,:,:,:] + 1j*y[1::2,:,:,:]
  Sc = np.zeros(yc.shape).astype(complex) # complex sensitivity maps
  S = np.zeros(y.shape) # real|image sensitivity maps
  for i in range(yc.shape[0]):
    tmp = np.squeeze(yc[i,:,:,:])
    Sc[i,:,:,:], eig_val = app.EspiritCalib(tmp, calib_width=cw, thresh=0.02, kernel_width=6, crop=cp, max_iter=100, device=sp.cpu_device, output_eigenvalue=True, show_pbar=False).run()
  S[0::2,:,:,:] = np.real(Sc)
  S[1::2,:,:,:] = np.imag(Sc)

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(S,[1,2,0,3]), [n[0]*c,n[1]*m]), vmin=-1, vmax=1, cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()
  return S, Sc;

def add_noise(y, msk, Np, seed):
  print(y.shape)
  print(msk.shape)
  rng = np.random.default_rng(301*seed)
  ymax = np.max(np.abs(y))
  y = y + rng.normal(0, ymax*Np,(y.shape))
  for i in range (msk.shape[0]):
    y[2*i:2*i+2,:,:,:] = y[2*i:2*i+2,:,:,:] * msk[i,:,:,:]
  return y

def create_mask(N, n, Na, R, seed):
  rng = np.random.default_rng(101*seed)
  ctr = np.floor(n/2 + 1).astype(int) # image center
  kc = (n[1]/R - 2*Na)/(n[1] - 2*Na) # fraction to k-space lines keep
  msk = np.zeros([N,n[0],n[1]])
  for i in range(N):    
    msk[i, :, rng.choice(n[1], int(n[1]*kc), replace=False)] = 1
    msk[i, :, ctr[1]-Na:ctr[1]+Na] = 1

  print('Acceleration rate: ' ,'%1.2f' %(1/np.mean(msk)))

  return msk


def create_noisy(x, msk, N, n, Np, seed):
  rng = np.random.default_rng(201*seed)
  y =    np.zeros((2*N, n[0], n[1]))
  yn =   np.zeros((2*N, n[0], n[1]))
  yu =   np.zeros((2*N, n[0], n[1]))
  ynAbs= np.zeros((N, n[0], n[1]))

  print("x: ", np.min(takeMag(x)), np.max(takeMag(x)))

  for i in range(N):
    # print("x inside for loop: ", np.min(takeMag(x[2*i:2*i+2,:,:])), np.max(takeMag(x[2*i:2*i+2,:,:])))
    y[2*i:2*i+2,:,:] = torch_to_np(fft2c_ra(np_to_torch(x[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
    # print("y inside for loop: ", np.min(takeMag(y[2*i:2*i+2,:,:])), np.max(takeMag(y[2*i:2*i+2,:,:])))

    if i==0:
      ymax = np.abs(np.max(y))
    # print(rng.normal(0, ymax/10,(2*N, n[0], n[1])).shape)
    yn[2*i:2*i+2,:,:] = y[2*i:2*i+2,:,:] + rng.normal(0, ymax*Np,(2, n[0], n[1]))
    ynAbs[i,:,:] = np.sqrt(np.sum((np.abs(yn[2*i:2*i+2,:,:]))**2, axis=0))
    # ynAbs[i,:,:] = np.abs((ynAbs[i,:,:]/np.max(ynAbs[i,:,:])))**0.25
    yu[2*i:2*i+2,:,:] = yn[2*i:2*i+2,:,:]*msk[i:i+1,:,:]

  print("y: ", np.min(takeMag(y)), np.max(takeMag(y)))
  print("yn: ", np.min(takeMag(yn)), np.max(takeMag(yn)))
  print("yu: ", np.min(takeMag(yu)), np.max(takeMag(yu)))


  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  tmp = np.concatenate(( torch_to_np(ifft2c_ra(np_to_torch(y[0:2,:,:]))), torch_to_np(ifft2c_ra(np_to_torch(yn[0:2,:,:]))), torch_to_np(ifft2c_ra(np_to_torch(yu[0:2,:,:]))) ), axis=2)
  tmp = np.reshape(np.transpose(tmp,[1,0,2]), [n[0],n[1]*6])
  # print(tmp.shape)

  plt.imshow(tmp[:,:], cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(ynAbs**0.25,[1,0,2]), [n[0],n[1]*N]), cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(msk,[1,0,2]), [n[0],n[1]*N]), cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,8),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose((ynAbs**0.25)*msk,[1,0,2]), [n[0],n[1]*N]), cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  # # Delete unwanted variable
  # # !nvidia-smi
  # del yn, y, tmp
  # torch.cuda.empty_cache()
  # # !nvidia-smi

  return y, yn, yu;


def create_noisy_SNR(x, msk, N, n, snr, seed):
  rng = np.random.default_rng(201*seed)
  y =    np.zeros((2*N, n[0], n[1]))
  yn =   np.zeros((2*N, n[0], n[1]))
  yu =   np.zeros((2*N, n[0], n[1]))
  ynAbs= np.zeros((N, n[0], n[1]))
#
  #psnr = np.zeros(N)

  for i in range(N):
    y[2*i:2*i+2,:,:] = torch_to_np(fft2c_ra(np_to_torch(x[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
    #if i==0:
    #ymax = np.abs(np.max(y))
    # print(rng.normal(0, ymax/10,(2*N, n[0], n[1])).shape)
    
    ## from snr to variance
    var = ((np.linalg.norm(y[2*i:2*i+2,:,:]))**2) / ((n[0]*n[1])*10**(snr/10))

    
    
    yn[2*i:2*i+2,:,:] = y[2*i:2*i+2,:,:] + rng.normal(0, var**(1/2),(2, n[0], n[1]))
    ynAbs[i,:,:] = np.sqrt(np.sum((np.abs(yn[2*i:2*i+2,:,:]))**2, axis=0))
    # ynAbs[i,:,:] = np.abs((ynAbs[i,:,:]/np.max(ynAbs[i,:,:])))**0.25
    yu[2*i:2*i+2,:,:] = yn[2*i:2*i+2,:,:]*msk[i:i+1,:,:]
  print("Noise Var: ", var)
 
    #PSNR
#     original = y
#     compressed = yn
#     mse = np.mean((original - compressed) ** 2)
#     if(mse == 0):  # MSE is zero means no noise is present in the signal .
#                   # Therefore PSNR have no importance.
#         mse = 1/1e10
#     max_pixel = ymax #255.0
#     psnr[i] = 20 * log10(max_pixel / sqrt(mse))
    
#   mpsnr = np.mean(psnr) 
#   print("Mean PSNR (dB): ", mpsnr)

  fig = plt.figure(figsize=(16,5),facecolor='white', edgecolor=None)
  tmp = np.concatenate(( torch_to_np(ifft2c_ra(np_to_torch(y[0:2,:,:]))), torch_to_np(ifft2c_ra(np_to_torch(yn[0:2,:,:]))), torch_to_np(ifft2c_ra(np_to_torch(yu[0:2,:,:]))) ), axis=2)
  tmp = np.reshape(np.transpose(tmp,[1,0,2]), [n[0],n[1]*6])
  # print(tmp.shape)

  plt.imshow(tmp[:,:], cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,5),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(ynAbs**0.25,[1,0,2]), [n[0],n[1]*N]), cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,5),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose(msk,[1,0,2]), [n[0],n[1]*N]), cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  fig = plt.figure(figsize=(16,5),facecolor='white', edgecolor=None)
  plt.imshow(np.reshape(np.transpose((ynAbs**0.25)*msk,[1,0,2]), [n[0],n[1]*N]), cmap=plt.cm.Greys_r) # use a specific color map
  plt.show()

  # # Delete unwanted variable
  # # !nvidia-smi
  # del yn, y, tmp
  # torch.cuda.empty_cache()
  # # !nvidia-smi

  return y, yn, yu;

# """
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# """

# from typing import List, Optional

# import torch
# import torch.fft


# # ra: my implmentation of 2D FFT assumes the real/imag dimension to be the second
# def fft2c_ra(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
#     """
#     Apply centered 2 dimensional Fast Fourier Transform.
#     Args:
#         data: Complex valued input data containing at least 3 dimensions:
#             dimensions -3 & -2 are spatial dimensions and dimension -1 has size
#             2. All other dimensions are assumed to be batch dimensions.
#         norm: Normalization mode. See ``torch.fft.fft``.
#     Returns:
#         The FFT of the input.
#     """
#     # print('FFT: ', data.shape)
#     data = torch.permute(data, (0,2,3,1)) # ra: added
#     if not data.shape[-1] == 2:
#         raise ValueError("Tensor does not have separate complex dim.")

#     data = ifftshift(data, dim=[-3, -2])
#     data = torch.view_as_real(
#         torch.fft.fftn(  # type: ignore
#             torch.view_as_complex(data), dim=(-2, -1), norm=norm
#         )
#     )
#     data = fftshift(data, dim=[-3, -2])
#     data = torch.permute(data, (0,3,1,2)) # ra: added
#     return data


# def ifft2c_ra(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
#     """
#     Apply centered 2-dimensional Inverse Fast Fourier Transform.
#     Args:
#         data: Complex valued input data containing at least 3 dimensions:
#             dimensions -3 & -2 are spatial dimensions and dimension -1 has size
#             2. All other dimensions are assumed to be batch dimensions.
#         norm: Normalization mode. See ``torch.fft.ifft``.
#     Returns:
#         The IFFT of the input.
#     """
#     # print('iFFT: ', data.shape)
#     data = torch.permute(data, (0,2,3,1)) # ra: added
#     if not data.shape[-1] == 2:
#         raise ValueError("Tensor does not have separate complex dim.")

#     data = ifftshift(data, dim=[-3, -2])
#     data = torch.view_as_real(
#         torch.fft.ifftn(  # type: ignore
#             torch.view_as_complex(data), dim=(-2, -1), norm=norm
#         )
#     )
#     data = fftshift(data, dim=[-3, -2])
#     data = torch.permute(data, (0,3,1,2)) # ra: added
#     return data


# def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
#     """
#     Apply centered 2 dimensional Fast Fourier Transform.
#     Args:
#         data: Complex valued input data containing at least 3 dimensions:
#             dimensions -3 & -2 are spatial dimensions and dimension -1 has size
#             2. All other dimensions are assumed to be batch dimensions.
#         norm: Normalization mode. See ``torch.fft.fft``.
#     Returns:
#         The FFT of the input.
#     """
#     if not data.shape[-1] == 2:
#         raise ValueError("Tensor does not have separate complex dim.")

#     data = ifftshift(data, dim=[-3, -2])
#     data = torch.view_as_real(
#         torch.fft.fftn(  # type: ignore
#             torch.view_as_complex(data), dim=(-2, -1), norm=norm
#         )
#     )
#     data = fftshift(data, dim=[-3, -2])

#     return data


# def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
#     """
#     Apply centered 2-dimensional Inverse Fast Fourier Transform.
#     Args:
#         data: Complex valued input data containing at least 3 dimensions:
#             dimensions -3 & -2 are spatial dimensions and dimension -1 has size
#             2. All other dimensions are assumed to be batch dimensions.
#         norm: Normalization mode. See ``torch.fft.ifft``.
#     Returns:
#         The IFFT of the input.
#     """
#     if not data.shape[-1] == 2:
#         raise ValueError("Tensor does not have separate complex dim.")

#     data = ifftshift(data, dim=[-3, -2])
#     data = torch.view_as_real(
#         torch.fft.ifftn(  # type: ignore
#             torch.view_as_complex(data), dim=(-2, -1), norm=norm
#         )
#     )
#     data = fftshift(data, dim=[-3, -2])

#     return data


# # Helper functions


# def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
#     """
#     Similar to roll but for only one dim.
#     Args:
#         x: A PyTorch tensor.
#         shift: Amount to roll.
#         dim: Which dimension to roll.
#     Returns:
#         Rolled version of x.
#     """
#     shift = shift % x.size(dim)
#     if shift == 0:
#         return x

#     left = x.narrow(dim, 0, x.size(dim) - shift)
#     right = x.narrow(dim, x.size(dim) - shift, shift)

#     return torch.cat((right, left), dim=dim)


# def roll(
#     x: torch.Tensor,
#     shift: List[int],
#     dim: List[int],
# ) -> torch.Tensor:
#     """
#     Similar to np.roll but applies to PyTorch Tensors.
#     Args:
#         x: A PyTorch tensor.
#         shift: Amount to roll.
#         dim: Which dimension to roll.
#     Returns:
#         Rolled version of x.
#     """
#     if len(shift) != len(dim):
#         raise ValueError("len(shift) must match len(dim)")

#     for (s, d) in zip(shift, dim):
#         x = roll_one_dim(x, s, d)

#     return x


# def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
#     """
#     Similar to np.fft.fftshift but applies to PyTorch Tensors
#     Args:
#         x: A PyTorch tensor.
#         dim: Which dimension to fftshift.
#     Returns:
#         fftshifted version of x.
#     """
#     if dim is None:
#         # this weird code is necessary for toch.jit.script typing
#         dim = [0] * (x.dim())
#         for i in range(1, x.dim()):
#             dim[i] = i

#     # also necessary for torch.jit.script
#     shift = [0] * len(dim)
#     for i, dim_num in enumerate(dim):
#         shift[i] = x.shape[dim_num] // 2

#     return roll(x, shift, dim)


# def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
#     """
#     Similar to np.fft.ifftshift but applies to PyTorch Tensors
#     Args:
#         x: A PyTorch tensor.
#         dim: Which dimension to ifftshift.
#     Returns:
#         ifftshifted version of x.
#     """
#     if dim is None:
#         # this weird code is necessary for toch.jit.script typing
#         dim = [0] * (x.dim())
#         for i in range(1, x.dim()):
#             dim[i] = i

#     # also necessary for torch.jit.script
#     shift = [0] * len(dim)
#     for i, dim_num in enumerate(dim):
#         shift[i] = (x.shape[dim_num] + 1) // 2

#     return roll(x, shift, dim)