% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L+S reconstruction of undersampled multicoil cardiac cine MRI
%
% Ricardo Otazo (2013)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;close all;


% load undersampled data
data_path = "..\data\MRXCAT\";
folder = "with_lesion\";
patient = "male_pt77\";
path = data_path+folder+patient;

FS = 1;
if FS==0
    R=5.069307; %5.069307
else
    R=5;
end

path_r = path + "R"+num2str(R)+"\"+"comparison\";
% end



% load undersampled data 
formatSpec = '%.6f';
load(path_r+"yu"+"_R_"+num2str(R,formatSpec)+".mat");
kdata = permute(yu, [3, 4, 1, 2]);
[nx,ny,nt,nc]=size(kdata);
class(kdata)
% check scale: 
min(abs(kdata(:))), max(abs(kdata(:)))

load(data_path+"sen_maps_" +num2str(nx)+"_"+num2str(ny)+"_"+num2str(nc)+".mat")
b1=rot90(sen, 2);
sen_msk=ones(nx,ny);
% load mask
nIter=300;
tau=1e-6;
admm=1;
if FS==1
    if admm==1
        load(path+"xRef_N_"+num2str(nt)+"_ADMMnIters_"+num2str(nIter)+"_tau_"+num2str(tau,formatSpec)+"_again.mat");
    else
        load(path+"xRef_N_"+num2str(nt)+".mat");
    end      
    load(data_path+"mask_PE"+num2str(ny)+"_FR"+num2str(nt)+"_R"+num2str(R)+".mat");
    mask=permute(repmat(samp, [1, 1, nx]), [3, 1, 2]);
else
    load(path+folder+file+"mask_R_"+num2str(R,formatSpec)+".mat");
    mask=permute(squeeze(samp(:,1,:,:)), [2, 3, 1]);
end
size(mask)
class(mask)
% mask = permute(mask1(:,:,:,1), [3,1,2]);
% figure;imagesc(abs(mask(:,:,1)),[0,0.1*max(abs(mask(:)))]); axis off; axis image; colormap(gray);

%% ablation
ablation=0;
N=32; % select ablation
if ablation==1
    path_sv = path + "R"+num2str(R)+"\"+"ablation\frames\LplusS\";
else
    path_sv = path_r;
end
nt=N;
kdata = kdata(:,:,1:nt,:);
mask = mask(:,:,1:nt);
if FS==1
    mc = mc(1:nt,:,:);
end

% L+S reconstruction ******************************************************

param.E=Emat_xyt(mask,b1); % encoding matrix (forward operator): input: mask and sen maps
param.d=kdata; % sacle=1; given undersampled kspace measurement
param.T=TempFFT(3); % sparsifying transform: Temp FFT here

% correlated motion in the background
% param.lambda_L=0.0025;param.lambda_S=0.00125;
% stationary background

param.nite=200; % 50 % num iterations
param.tol=0.0001; % optimization stopping criteria: when update is too small

% training parameters: 
sv=1;
param.lambda_L=0.0025; % strength of Low-Rank % default 0.005
param.lambda_S=0.05; % strength of Sparsity % default 0.01

tStart = tic;

fprintf("R: %d\n", R);
[L,S] = lps_ist(param); % training function call

tEnd = toc(tStart);      % pair 2: toc
fprintf("Elasped time is: %f seconds\n", tEnd)

L = rot90(L, 2);
S = rot90(S, 2);

% L+S recon image:
LplusS=L+S; % recon image series: [nx, ny, nt]

% display 4 frames [2, 8, 14, 20]
if nt==32
    LplusSd=LplusS(:,:,2);LplusSd=cat(2,LplusSd,LplusS(:,:,8));LplusSd=cat(2,LplusSd,LplusS(:,:,14));LplusSd=cat(2,LplusSd,LplusS(:,:,20));
    Ld=L(:,:,2);Ld=cat(2,Ld,L(:,:,8));Ld=cat(2,Ld,L(:,:,14));Ld=cat(2,Ld,L(:,:,20));
    Sd=S(:,:,2);Sd=cat(2,Sd,S(:,:,8));Sd=cat(2,Sd,S(:,:,14));Sd=cat(2,Sd,S(:,:,20));
elseif nt==16
    LplusSd=LplusS(:,:,2);LplusSd=cat(2,LplusSd,LplusS(:,:,6));LplusSd=cat(2,LplusSd,LplusS(:,:,10));LplusSd=cat(2,LplusSd,LplusS(:,:,14));
    Ld=L(:,:,2);Ld=cat(2,Ld,L(:,:,6));Ld=cat(2,Ld,L(:,:,10));Ld=cat(2,Ld,L(:,:,14));
    Sd=S(:,:,2);Sd=cat(2,Sd,S(:,:,6));Sd=cat(2,Sd,S(:,:,10));Sd=cat(2,Sd,S(:,:,14));
elseif nt==8
    LplusSd=LplusS(:,:,2);LplusSd=cat(2,LplusSd,LplusS(:,:,4));LplusSd=cat(2,LplusSd,LplusS(:,:,6));LplusSd=cat(2,LplusSd,LplusS(:,:,8));
    Ld=L(:,:,2);Ld=cat(2,Ld,L(:,:,4));Ld=cat(2,Ld,L(:,:,6));Ld=cat(2,Ld,L(:,:,8));
    Sd=S(:,:,2);Sd=cat(2,Sd,S(:,:,4));Sd=cat(2,Sd,S(:,:,6));Sd=cat(2,Sd,S(:,:,8));
elseif nt==4
    LplusSd=LplusS(:,:,1);LplusSd=cat(2,LplusSd,LplusS(:,:,2));LplusSd=cat(2,LplusSd,LplusS(:,:,3));LplusSd=cat(2,LplusSd,LplusS(:,:,4));
    Ld=L(:,:,1);Ld=cat(2,Ld,L(:,:,2));Ld=cat(2,Ld,L(:,:,3));Ld=cat(2,Ld,L(:,:,4));
    Sd=S(:,:,1);Sd=cat(2,Sd,S(:,:,2));Sd=cat(2,Sd,S(:,:,3));Sd=cat(2,Sd,S(:,:,4));
end

clip=1;epsilon=1e-10;
figure;
subplot(3,1,1),imagesc(abs(LplusSd),[0,clip*max(abs(LplusSd(:)))]); axis off; axis image; colormap(gray);title('L+S');
subplot(3,1,2),imagesc(abs(Ld),[0,clip*max(abs(Ld(:)))]); axis off; axis image; colormap(gray);title('L');
subplot(3,1,3),imagesc(abs(Sd),[0,clip*max(abs(Sd(:)))+epsilon]); axis off; axis image; colormap(gray);title('S');

% results
LplusS = LplusS.*sen_msk;
xLS = permute(LplusS, [3, 1, 2]);
% shift adjust:
xLS = correct_shift_LplusS(xLS);
if sv==1
save(path_sv+'LplusS'+'_N_'+num2str(nt)+'_R_'+num2str(R, formatSpec)+'.mat', 'xLS');end
if FS==1
mc = permute(permute(mc, [2,3,1]).*sen_msk, [3,1,2]); 
xAbs = abs(mc); % (32, 160, 96)
% x = complex_to_real_plus_imag(mc);
xHat = complex_to_real_plus_imag(xLS);
xHatAbs = abs(xLS);
% calculate metrics: [xAbs, xHatAbs]
% nmse:
% % SLOW for loop...
% nmse = zeros(1,nt);
% for i=1:nt
%     nmse(1,i) = mean((x(i*2-1:i*2,:,:) - xHat(i*2-1:i*2,:,:)).^2, 'all') / mean(x(i*2-1:i*2,:,:).^2,'all'); 
% %     10*log10(nmse(1,i))
% %     ssim(1,i) = 
% 
% end
% nmse_mean = 10*log10(mean(nmse))

% % FAST :
nmse= mean(abs(mc - xLS).^2, [2,3]) ./ mean(abs(mc).^2, [2,3]);
nmse_mean = 10*log10(mean(nmse))

% % error maps:
if ablation==0
errMap = xAbs-xHatAbs;
i = 2; % 
offset= 6;
xAbs_ = xAbs(i:offset:20,:,:);
xAbsCat = cat(2,[squeeze(xAbs_(1,:,:)), squeeze(xAbs_(2,:,:)), squeeze(xAbs_(3,:,:)), squeeze(xAbs_(4,:,:))]) ;
xHatAbs_ = xHatAbs(i:offset:20,:,:);
xHatAbsCat = cat(2,[squeeze(xHatAbs_(1,:,:)), squeeze(xHatAbs_(2,:,:)), squeeze(xHatAbs_(3,:,:)), squeeze(xHatAbs_(4,:,:))]) ;
err = errMap(i:offset:20,:,:); % 4, nx, ny
errCat = cat(2,[squeeze(err(1,:,:)), squeeze(err(2,:,:)), squeeze(err(3,:,:)), squeeze(err(4,:,:))]) ;

figure;
subplot(3, 1, 1), imagesc(xAbsCat,[0,clip*max(xAbsCat(:))]); axis off; axis image; colormap(gray);title('xRef');
subplot(3, 1, 2), imagesc(xHatAbsCat,[0,clip*max(xHatAbsCat(:))]); axis off; axis image; colormap(gray);title('L+S Recon');
subplot(3, 1, 3), imagesc(errCat, [0,max(errMap(:))/8]); axis off; axis image;title('Error Maps');
end

end


function y = correct_shift_LplusS(x)
y0=zeros(size(x)); y0(:,:,1)=x(:,:,end); y0(:,:,2:end)=x(:,:,1:end-1);
y=zeros(size(x)); y(:,1,:)=y0(:,end,:); y(:,2:end,:)=y0(:,1:end-1,:);
end