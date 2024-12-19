import numpy as np
import matplotlib.pyplot as plt
from sigpy.mri import app
import torch

import torch.nn as nn
import torchvision
import sys
from PIL import Image
import PIL
import pywt
import time
from scipy import signal

from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr




# Just use np.transpose

# # reshape to convention (N, Nc, x, y)
# def reshape_to_N_Nc_Nxy(kspace):
    
#     print("\nInside reshape")
#     ks_size = np.shape(kspace)
#     N = ks_size[4]
#     c = ks_size[3]
#     kspace_coil = kspace[:, :, 0, :, :]

#     # for loop
#     kspace_coil_reshaped = np.zeros((ks_size[4], ks_size[3], ks_size[0], ks_size[1])).astype(complex)
#     for i in range(N):
#         for j in range(c):
#             kspace_coil_reshaped[i, j, :, :] = kspace_coil[:,:, j, i]

#     print(np.shape(kspace_coil_reshaped))    
#     print(kspace_coil_reshaped.dtype)
#     return kspace_coil_reshaped



# zero-padding to the next multiple of 64
def zero_padding(kspace, Nm):
    
    print("\nInside zero-pading")

    # original size:
    ni = np.shape(kspace)[2:4]
    print(ni)
    
    # Round up to 64 multiple:
    (n) = (np.ceil(np.float32(ni)/Nm)*Nm).astype(int) # Round up to an even-sized array
    print(n)
    
    # zero-pad:
    pxy = n-ni
    (px, py) = (np.floor(pxy/2)).astype(int)
    
    kspace_coil_reshaped_padded = np.pad(kspace, ((0,0), (0,0), (px,px), (py,py))) 
    print(np.shape(kspace_coil_reshaped_padded))  
    print(kspace_coil_reshaped_padded.dtype)
#     plt.imshow(np.abs(kspace_coil_reshaped_padded[0, 0,:,:])**0.1) # 2D image
    
    return kspace_coil_reshaped_padded


# convert complex to real+imag
def complex_to_real_plus_imag_4d(x):
    print("\nInside complex_to_real_plus_imag")

    (N, Nc, nx, ny) = np.shape(x)
    yuN = np.zeros((2*N, Nc, nx, ny))
    yuN[0::2,:,:,:] = np.real(x)
    yuN[1::2,:,:,:] = np.imag(x)
    print(np.shape(yuN))
    print(yuN.dtype)

    return yuN
# convert complex to real+imag
def complex_to_real_plus_imag_3d(x):
    print("\nInside complex_to_real_plus_imag")

    (N, nx, ny) = np.shape(x)
    yuN = np.zeros((2*N, nx, ny))
    yuN[0::2,:,:] = np.real(x)
    yuN[1::2,:,:] = np.imag(x)
    print(np.shape(yuN))
    print(yuN.dtype)

    return yuN


# convert complex to real+imag
def real_plus_imag_to_complex_4d(x):
#     print("\nInside real_plus_imag_to_complex")

    (M, Nc, nx, ny) = np.shape(x) # M=2N
    yuN = np.zeros((int(M/2), Nc, nx, ny))
    yuN = x[0::2,:,:,:] + 1j*x[1::2,:,:,:] 
#     print(np.shape(yuN))
#     print(yuN.dtype)

    return yuN

# convert complex to real+imag
def real_plus_imag_to_complex_3d(x):
#     print("\nInside real_plus_imag_to_complex")

    (M, nx, ny) = np.shape(x) # M=2N
    yuN = np.zeros((int(M/2), nx, ny))
    yuN = x[0::2,:,:] + 1j*x[1::2,:,:] 
#     print(np.shape(yuN))
#     print(yuN.dtype)

    return yuN


def scale(yuN, ksp_scale):
    print("\nInside Scale")
    ymax = np.max(np.sqrt(yuN[0::2,:,:,:]**2 + yuN[1::2,:,:,:]**2))
    print(ymax)
    
    yuN = ksp_scale*yuN/ymax
    yuNmax = np.max(np.sqrt(yuN[0::2,:,:,:]**2 + yuN[1::2,:,:,:]**2))
    print(yuNmax)
    
    return yuN



def read_myData(kspace, zp, Nm, N, ksp_scale):
    # kspace: (Nx, Ny, Nc, N)
    # change size to convention (N, Nc, Nx, Ny):
    #kspace_coil_reshaped = reshape_to_N_Nc_Nxy(kspace)
    kspace_coil_reshaped = np.transpose(kspace, (3, 2, 0, 1))


    # zero-pad to the next Nm multiple
    if (zp):
        kspace_coil_reshaped_padded = zero_padding(kspace_coil_reshaped, Nm)
    else:
        kspace_coil_reshaped_padded = kspace_coil_reshaped
            

    ## prepare for network input:

    # convert complex to real+imag
    ksp_ = kspace_coil_reshaped_padded[0:N,:,:,:]
    k_max = np.max(np.abs(ksp_))
    ksp = ksp_scale*ksp_/k_max

   # ksp = scale(ksp,ksp_scale)
    yu0 =   complex_to_real_plus_imag(ksp_)

    # scale kspace
    yuN = scale(yu0,ksp_scale)

    print("\nData ready: ")
    print(np.shape(yuN))
    print(yuN.dtype)
    #print(np.max(yuN))
    
    return yuN, ksp

    # yuN ready for DISCUS
    

def snr_estimate_ksp(data, lines):
    # Estimates SNR from the acquired kspace data:
    # data: kspace data.... [2*N, CH, RO, PE] real+imag
    # lines: Number of bottom RO lines to estimate noise power from

    # estimate noise variance:
    portion = data[:,:,-lines:, :] # data portion for noise estimation
    print(portion.shape)
    var_n = np.var(portion)
    print(var_n)
    # calculate SNR in dB
    snr_db = 10*np.log10(((np.linalg.norm(data)**2)/(data.size)) / (var_n+1e-10))
    print(snr_db)

    return snr_db


    
def mask_create(y):
    msk = (y!=0)
    print(np.shape(msk))
    #mskint = int(msk)

    #mskint = list(map(int, msk))

    mskint = msk*1
    #print(mskint)
    plt.imshow(mskint[0,0,:,:])
    print(np.min(mskint))
    print(np.max(mskint))

    # acceleration rate
    non_zero = np.count_nonzero(mskint)
    zeros_ = np.count_nonzero(mskint==0)
    total = non_zero + zeros_
    perc = (non_zero/total)*100
    print(perc, 100/perc) # acc rate
    
    return mskint
    

    
def sense_ESPIRiT(ksp):
    # ksp: (N, Nc, Nx, Ny)

    (N, Nc, kx, ky) = np.shape(ksp)
    maps = np.zeros((N, Nc, kx, ky)).astype(complex)

    for i in range(N):
        maps[i,:,:,:] = app.EspiritCalib(ksp[i, :,:,:], calib_width=24,
                         thresh=0.02, kernel_width=6, crop=0.95,
                         max_iter=100,
                         output_eigenvalue=False, show_pbar=True).run()

    #print(maps)
    print(np.shape(maps))
    # visualize first frame maps...

    #sense_reshaped = maps
    
    
    # maps: (2, 4, 192, 192) complex

    plt.figure(figsize=(20, 8)) # (width, height) in inches

    for i in range(Nc): # no. of images
        ax = plt.subplot(1, Nc, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p = np.power(np.abs(maps[0,i,:,:]), 1/3)
        plt.imshow(p, cmap='gray', vmin = 0, vmax = 1) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()
    
    # Initialise the subplot function using number of rows and columns
    #figure, axis = plt.subplots(Nc)
    
#     for i in range(Nc):
#         p = np.power(np.abs(sense_reshaped[0,i,:,:]), 1/3)
#         plt.imshow(p, cmap='gray')
#         plt.show()

    # Combine all the operations and display
    
    
    
#     plt.imshow(np.power(np.abs(sense_reshaped[0,0,:,:]), 1/3), cmap='gray') # 2D image
#     plt.show()

#     plt.imshow(np.power(np.abs(sense_reshaped[0,1,:,:]), 1/3), cmap='gray') # 2D image
#     plt.show()

#     plt.imshow(np.power(np.abs(sense_reshaped[0,2,:,:]), 1/3), cmap='gray') # 2D image
#     plt.show()

#     plt.imshow(np.power(np.abs(sense_reshaped[0,3,:,:]), 1/3), cmap='gray') # 2D image
#     plt.show()
    return maps



## sense maps for FS ref:
def sense_ESPIRiT_FS_kspace(ksp, calib_width=24, kernel_width=6, crop=0.8):
# ksp: FS ksp: (N, Nc, kx, ky) or (FR, CH, RO, PE)
# calib_width: ACS width
# kernel_width: 

# maps: (Nc, kx, ky) or (CH, RO, PE)

    (N, Nc, kx, ky) = np.shape(ksp)
    maps = np.zeros((Nc, kx, ky)).astype(complex)

    #ksp:
    # ksp_espirit = np.sum(ksp, axis = 0) / (np.sum(mskN, axis = 0) + epsilon) 
    ksp_espirit = np.sum(ksp, axis = 0) / N # N: no. of frames 
    # no division by mask is required (FS)

    #for i in range(N):
    maps = app.EspiritCalib(ksp_espirit, calib_width=calib_width,
                     thresh=0.02, kernel_width=kernel_width, crop=crop,
                     max_iter=100,
                     output_eigenvalue=False, show_pbar=True).run()

    #print(maps)
    print("ESPIRiT Generated Maps Size: ", np.shape(maps))
    
    ## check sense maps across coil dim:

    plt.figure(figsize=(20, 8)) # (width, height) in inches

    for i in range(Nc): # no. of images
        ax = plt.subplot(1, Nc, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p = np.power(np.abs(maps[i,:,:]), 1)
        plt.imshow(p, cmap='gray', vmin = 0, vmax = 1) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()

    return maps




## sense maps for FS ref:
def MRXCAT_ESPIRiT_FS_kspace(ksp, calib_width=24, kernel_width=6, crop=0.95):
# ksp: FS ksp: (N, Nc, kx, ky) or (FR, CH, RO, PE)
# calib_width: ACS width
# kernel_width: 

# maps: (Nc, kx, ky) or (CH, RO, PE)

    (N, Nc, kx, ky) = np.shape(ksp)
    maps = np.zeros((N, Nc, kx, ky)).astype(complex)

    #ksp:
    # ksp_espirit = np.sum(ksp, axis = 0) / (np.sum(mskN, axis = 0) + epsilon) 
#     ksp_espirit = np.sum(ksp, axis = 0) / N # N: no. of frames 
    # no division by mask is required (FS)

    for i in range(N):
        maps[i, :,:,:] = app.EspiritCalib(ksp[i,:,:,:], calib_width=calib_width,
                     thresh=0.02, kernel_width=kernel_width, crop=crop,
                     max_iter=100,
                     output_eigenvalue=False, show_pbar=True).run()

    #print(maps)
    print("ESPIRiT Generated Maps Size: ", np.shape(maps))
    
    ## check sense maps across coil dim:

    plt.figure(figsize=(20, 8)) # (width, height) in inches

    for i in range(Nc): # no. of images
        ax = plt.subplot(1, Nc, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p = np.power(np.abs(maps[0,i,:,:]), 1)
        plt.imshow(p, cmap='gray', vmin = 0, vmax = 1) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()

    return maps






## sense maps for FS ref:
def sense_ESPIRiT_GRO(ksp, mskN, calib_width=24, kernel_width=6, crop=0.8):
# ksp: FS ksp: (N, Nc, kx, ky) or (FR, CH, RO, PE)
# calib_width: ACS width
# kernel_width: 

# maps: (Nc, kx, ky) or (CH, RO, PE)

    (N, Nc, kx, ky) = np.shape(ksp)
    maps = np.zeros((Nc, kx, ky)).astype(complex)
    
    #ksp:
    epsilon = 1e-10
    ksp_espirit = np.sum(ksp, axis = 0) / (np.sum(mskN, axis = 0) + epsilon)

    
    maps = app.EspiritCalib(ksp_espirit, calib_width=24,
                     thresh=0.02, kernel_width=6, crop=crop,
                     max_iter=100,
                     output_eigenvalue=False, show_pbar=True).run()

    #print(maps)
    print("ESPIRiT Generated Maps Size: ", np.shape(maps))
    # visualize first frame maps...

    #sense_reshaped = maps
    
    
    # maps: (2, 4, 192, 192) complex

    plt.figure(figsize=(20, 8)) # (width, height) in inches
    print("Displaying", Nc, "Coil Images:")


    for i in range(Nc): # no. of images
        ax = plt.subplot(1, Nc, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p = np.power(np.abs(maps[i,:,:]), 1)
        plt.imshow(p, cmap='gray', vmin = 0, vmax = 1) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()
    
    
    ## check sampl sum across coils

    plt.figure(figsize=(20, 8)) # (width, height) in inches
    print("Displaying", Nc, "Sampling Sum across coils:")

    for i in range(Nc): # no. of images
        ax = plt.subplot(1, Nc, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p = np.sum(mskN, axis = 0)
        plt.imshow(p[i, :,:]**0.6, cmap=plt.cm.Greys_r) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()


    
#     ## check ACS kspace across coil dim:

#     plt.figure(figsize=(20, 8)) # (width, height) in inches
#     print("Displaying ", Nc, "ACS kspace  across coils:")

#     for i in range(Nc): # no. of images
#         ax = plt.subplot(1, Nc, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
#         ax.tick_params(axis='both', colors='black') # show ticks on axes
#         p = np.abs(ksp_espirit[i,:,:])
#         plt.imshow(p, cmap='gray') # color map and mapping range
#         #plt.ylabel("accuracy", color = 'white')
#         #plt.xlabel("epoch", color = 'white')
#         #plt.legend(['train', 'val'], loc = 'upper left')

#     #plt.title('ESP
#     plt.show()



    return maps




def multc(x,y):
    x = torch.permute(x, (0,2,3,4,1)).contiguous()
    y = torch.permute(y, (0,2,3,4,1)).contiguous()
    
    x = torch.view_as_complex(x)
    y = torch.view_as_complex(y)
    
    z = x*y
    
    z = torch.view_as_real(z)
    z = torch.permute(z, (0,4,1,2,3))
    
    return z



## coil-combined for FS data:# reference images: fully sampled
def coil_combine(img, Sc):
# FS img: (N, Nc, kx, ky) or (FR, CH, RO, PE)
# Sc: sensitivity complex: (N, Nc, kx, ky)

# x: coil combined image: (N, kx, ky) or (FR, Nx, Ny)

#     size = np.shape(ksp)
#     N = size[0]
#     n = size[2:]

#     x = np.zeros((N, n[0], n[1])).astype(complex)        
#     for i in range(N):
        
    x = np.squeeze(np.sum(np.conjugate(Sc) * img, axis=1))        
     
    print("SENSE Coil combined Frames Size: ", x.shape)
    ## visualize images:
    # frames (no coils)


    plt.figure(figsize=(20, 8)) # (width, height) in inches
    N=4
    print("Displaying first", N, "Frames:")

    for i in range(N): # no. of images
        ax = plt.subplot(1, N, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p = abs(x[i,:,:]) # first coil
        plt.imshow(p**0.4, cmap=plt.cm.Greys_r) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()

    return x



  
## SENSE for FS data:# reference images: fully sampled
def SENSE_FS(ksp, Sc):
# FS ksp: (N, Nc, kx, ky) or (FR, CH, RO, PE)
# Sc: sensitivity complex: (Nc, kx, ky)

# x: coil combined image: (N, kx, ky) or (FR, Nx, Ny)

#     size = np.shape(ksp)
#     N = size[0]
#     n = size[2:]

#     x = np.zeros((N, n[0], n[1])).astype(complex)        
#     for i in range(N):
        
    x = pAt_FS(ksp,Sc) # both Sc and ksp are complex (single channel)
        
     
    print("SENSE Coil combined Frames Size: ", x.shape)
    ## visualize images:
    # frames (no coils)


    plt.figure(figsize=(20, 8)) # (width, height) in inches
    N=4
    print("Displaying first", N, "Frames:")

    for i in range(N): # no. of images
        ax = plt.subplot(1, N, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p = abs(x[i,:,:]) # first coil
        plt.imshow(p**0.4, cmap=plt.cm.Greys_r) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()

    return x


## Multicoil FS Data
def pAt_FS(y,Sc): # adjoint of the forward operator
# y: FS kspace: (N, Nc, kx, ky) or (FR, CH, RO, PE)
# Sc: sensitivity complex: (Nc, kx, ky)

# x: coil combined image: (N, kx, ky) or (N, Nx, Ny):  frames

  x = np.squeeze(np.sum(np.conjugate(Sc) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y, axes=(-2,-1)),norm='ortho', axes=(-2,-1)), axes=(-2,-1)),axis=1)) #torch_to_np(ifft2c_ra(np_to_torch(y[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
  return x




## Reshape GRO generated mask from MATLAB:
def reshape_GRO_samp(mask):
# mask: (PE, FR, RO, CH)

# msk: (FR, CH, RO, PE) as kspace

    msk = np.transpose(mask, (1, 3, 2, 0))

#     ks_size = np.shape(mask)
#     Ni = ks_size[1]
#     c = ks_size[3]

#     # for loop
#     msk = np.zeros((Ni, c, ks_size[2], ks_size[0]))
#     for i in range(Ni):
#         for j in range(c):
#             msk[i, j, :, :] = np.transpose(mask[:,i, :,j])

    print("Reshaped mask size:", np.shape(msk))    
    print("Reshaped mask dtype:", msk.dtype)

    ## masks generated:
    # across frames

    plt.figure(figsize=(20, 8)) # (width, height) in inches
    N=4
    print("Displaying first", N, "Frames (of first coil):")
    for i in range(N): # no. of images
        ax = plt.subplot(1, N, i + 1) # (Nx, Ny, i): total Nx x Ny images, current image no. i
        ax.tick_params(axis='both', colors='black') # show ticks on axes
        p =  msk[i, 0, :, :] # first coil
        plt.imshow(p**0.6, cmap=plt.cm.Greys_r) # color map and mapping range
        #plt.ylabel("accuracy", color = 'white')
        #plt.xlabel("epoch", color = 'white')
        #plt.legend(['train', 'val'], loc = 'upper left')

    #plt.title('ESP
    plt.show()

    return msk
    
  

# Returns magnitude images when real/imag and interleaved even/odd along axis=0
def takeMag(x): 
# x: (2*n, x, y)
# xMag: (n, x, y)
  xMag = np.zeros((int(x.shape[0]/2), x.shape[1], x.shape[2]))
  for i in range(xMag.shape[0]):
    xMag[i,:,:] = np.sqrt(x[2*i,:,:]**2 + x[2*i+1,:,:]**2)
  return xMag
# np.abs() when x is complex and of size (n, x, y) i.e., no separate real + imag channels






## CS Recon:

def pA(x,msk,Sc): # forward operator
  y = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Sc*x, axes=(-2,-1)),norm='ortho', axes=(-2,-1)), axes=(-2,-1)) # torch_to_np(fft2c_ra(np_to_torch(x[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
  y = y*msk
  return y

def pAt(y,msk,Sc): # adjoint of the forward operator
  y = y*msk
  x = np.squeeze(np.sum(np.conjugate(Sc) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y, axes=(-2,-1)),norm='ortho', axes=(-2,-1)), axes=(-2,-1)),axis=0)) #torch_to_np(ifft2c_ra(np_to_torch(y[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
  return x

def pAt_4d(y,msk,Sc): # adjoint of the forward operator
  y = y*msk
  x = np.squeeze(np.sum(np.conjugate(Sc) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y, axes=(-2,-1)),norm='ortho', axes=(-2,-1)), axes=(-2,-1)),axis=1)) #torch_to_np(ifft2c_ra(np_to_torch(y[2*i:2*i+2,:,:]), 'ortho')) # ra: correct?
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
    if i==0: # less sparsity strength for 1st band which is low-low
      coef[i,:,:] = np.maximum(np.abs(coef[i,:,:]) - tau*1e-1, 0) * np.exp(1j*np.angle(coef[i,:,:]))
    else:
      coef[i,:,:] = np.maximum(np.abs(coef[i,:,:]) - tau, 0) * np.exp(1j*np.angle(coef[i,:,:]))
  return coef

def admm_l1(x, y, msk, nIter, ss, mu, tau):
  B = 4
  u = At(y, msk)
  d = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  b = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  for i in range(nIter[0]):
    if i==0 or (i+1) % 10 ==0:
      print('Iteration: %2d ' %(i+1))
    for j in range(nIter[1]): #ra: hardcoded?
      gradA = At(A(u, msk) - y, msk)
      gradW = mu * iswt2_haar(swt2_haar(u) - d + b)
      u = u - ss * (gradA + gradW)
    d = st(swt2_haar(u) + b, tau/mu)
    b = b + (swt2_haar(u) - d)
  return u

def admm_pmri_l1(x, y, msk, Sc, nIter, ss, mu, tau):
  B = 4 # wavelet bands
  u = pAt(y, msk, Sc)
  d = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  b = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  for i in range(nIter[0]): # outer iter
    if i==0 or (i+1) % 10 ==0:
      print('Iteration: %2d ' %(i+1))
    for j in range(nIter[1]): # inner iter
      gradA = pAt(pA(u, msk, Sc) - y, msk, Sc)
      gradW = mu * iswt2_haar(swt2_haar(u) - d + b) 
      u = u - ss * (gradA + gradW) # grad. update
    # recon weights learnt: d, b
    d = st(swt2_haar(u) + b, tau/mu) 
    b = b + (swt2_haar(u) - d)
  return u # recon image


def admm_pmri_without_swt(y, msk, Sc, nIter, ss):
  # B = 4 # wavelet bands
  u = pAt(y, msk, Sc)
  # d = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  # b = np.zeros((B,x.shape[0],x.shape[1])).astype(complex)
  for i in range(nIter[0]): # outer iter
    if i==0 or (i+1) % 10 ==0:
      print('Iteration: %2d ' %(i+1))
    for j in range(nIter[1]): # inner iter
      gradA = pAt(pA(u, msk, Sc) - y, msk, Sc)
      #gradW = mu * iswt2_haar(swt2_haar(u) - d + b)
      u = u - ss * gradA
    # recon weights learnt: d, b
  #   d = st(swt2_haar(u) + b, tau/mu) 
  #   b = b + (swt2_haar(u) - d)
  return u # recon image


# using xHat range 
def calc_ssim(xL1Mag, xRefMag, RO_crop=False): #abs. img
    (N, Nx) = xL1Mag.shape[0:2]
    if RO_crop:
      RO_offset = int(0.2*Nx)
    ssimL1 = np.zeros((N,1))
    for i in range(N):
        if RO_crop:
          ssimL1[i] = ssim(xL1Mag[i,RO_offset:-RO_offset,:], xRefMag[i,RO_offset:-RO_offset,:], data_range = xL1Mag[i,RO_offset:-RO_offset,:].max() - xL1Mag[i,RO_offset:-RO_offset,:].min()) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
        else:
          ssimL1[i] = ssim(xL1Mag[i,:,:], xRefMag[i,:,:], data_range = xL1Mag[i,:,:].max() - xL1Mag[i,:,:].min()) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
    # print(ssimL1.shape)
    # print(ssimL1)
    return ssimL1

def calc_ssim_without_range(xL1Mag, xRefMag):
    N = xL1Mag.shape[0]
    ssimL1 = np.zeros((N,1))
    for i in range(N):
        ssimL1[i] = ssim(xL1Mag[i,:,:], xRefMag[i,:,:]) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
    # print(ssimL1.shape)
    # print(ssimL1)
    return ssimL1

def calc_ssim_with_RefRange(xL1Mag, xRefMag): # sklearn: data_range = xRefMag[i,:,:].max() - xRefMag[i,:,:].min()
    N = xL1Mag.shape[0]
    ssimL1 = np.zeros((N,1))
    for i in range(N):
        ssimL1[i] = ssim(xL1Mag[i,:,:], xRefMag[i,:,:], data_range = xRefMag[i,:,:].max() - xRefMag[i,:,:].min()) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
    # print(ssimL1.shape)
    # print(ssimL1)
    return ssimL1

def calc_psnr_mag(xL1Mag, xRefMag): # abs. images
    N = xL1Mag.shape[0]
    psnrL1 = np.zeros((N,1))
    for i in range(N):
        psnrL1[i] = psnr(xRefMag[i,:,:], xL1Mag[i,:,:], data_range = xL1Mag[i,:,:].max() - xL1Mag[i,:,:].min()) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
    # print(ssimL1.shape)
    # print(ssimL1)
    return psnrL1

# fully vectorized
def calc_psnr(xL1, xRef, RO_crop=False): # complex images
    (N, Nx) = xL1.shape[0:2]
    if RO_crop:
      RO_offset = int(0.2*Nx)    
    psnrL1 = np.zeros((N,1))
    # for i in range(N):
    if RO_crop:
      psnrL1 = np.max(np.abs(xRef[:,RO_offset:-RO_offset,:]), axis=(-1,-2))**2 / np.mean(np.abs(xRef[:,RO_offset:-RO_offset,:]-xL1[:,RO_offset:-RO_offset,:])**2, axis=(-1,-2)) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
    else:
      psnrL1 = np.max(np.abs(xRef), axis=(-1,-2))**2 / np.mean(np.abs(xRef-xL1)**2, axis=(-1,-2)) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 

    # print(ssimL1.shape)
    # print(ssimL1)
    return psnrL1

def calc_nmse_Forloop(xL1, xRef): # either complex or real+imag but not absolute
    N = xL1.shape[0]
    nmseL1 = np.zeros((N,1))
    for i in range(N):
        nmseL1[i] = np.mean((xRef[i*2:(i+1)*2,:,:]-xL1[i*2:(i+1)*2,:,:])**2) / np.mean((xRef[i*2:(i+1)*2,:,:])**2) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
    # print(ssimL1.shape)
    # print(ssimL1)
    return nmseL1

# fully vectorized
def calc_nmse(xL1, xRef, RO_crop=False): # complex 
    (N,Nx) = xL1.shape[0:2]
    if RO_crop:
      RO_offset = int(0.2*Nx)    
    nmseL1 = np.zeros((N,1))
    # for i in range(N):
    if RO_crop:
      nmseL1 = np.mean(np.abs(xRef[:,RO_offset:-RO_offset,:]-xL1[:,RO_offset:-RO_offset,:])**2, axis=(-1,-2)) / np.mean(np.abs(xRef[:,RO_offset:-RO_offset,:])**2, axis=(-1,-2)) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min() 
    else:
      nmseL1 = np.mean(np.abs(xRef-xL1)**2, axis=(-1,-2)) / np.mean(np.abs(xRef)**2, axis=(-1,-2)) # xHatL1Abs[i:i+1,:,:].max() - xHatL1Abs[i:i+1,:,:].min()        
    # print(ssimL1.shape)
    # print(ssimL1)
    return nmseL1

