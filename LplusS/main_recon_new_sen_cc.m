% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L+S reconstruction of undersampled multicoil cardiac cine MRI
%
% Ricardo Otazo (2013)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;close all;

FS = 1;
id = "JBH_3CH"; %"JGR_3CH"; %SJK_SAX_MOCO %SJK_SAX_BASE
N=16; % select ablation

if id=="JGR_SAX_BASE" || id=="JGR_3CH"
   crop = 0.6; % old processed
elseif id=="GRH_3CH"
   crop=0.8;
elseif id=="JBH_3CH"
   crop=0.7;
   if FS==0
       crop=0.8;end
else
   crop = 0.9; % good for SJK
end

if FS==0
    R=5.069307; %5.069307
    if id=="JGR_SAX_BASE" || id=="JGR_3CH" % due to partial fourier kspace region 
       R=6.336634;
    elseif id=="GRH_3CH"
       R=7.603960; 
    end
else
    R=4; % iterate over different Rs
end



if strncmp(id,"SJK",3)
   folder = "20230125_CS_LGE_SKJ/"; %"20230125_CS_LGE_SKJ/"
   if FS==0
      if id=="SJK_SAX_BASE"
         file = "/meas_MID00257_FID117434_SS_TRUFI_CS_PSIR_SAX_BASE_SAX/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      elseif id=="SJK_SAX_MOCO"
         file = "/meas_MID00258_FID117435_SS_TRUFI_CS_PSIR_SAX_SAX_MOCO_54_6/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      elseif id=="SJK_3CH"
         file = "/meas_MID00259_FID117436_SS_TRUFI_CS_PSIR_SAX_3CH/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      end
   else
      if id=="SJK_SAX_BASE"
         file = "/meas_MID00260_FID117437_SS_TRUFI_PSIR_SAX_FULL_SAMP_BASE_SAX/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      elseif id=="SJK_SAX_MOCO"
         file = "/meas_MID00261_FID117438_SS_TRUFI_PSIR_SAX_FULL_SAMP_SAX_MOCO_54_6/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      elseif id=="SJK_3CH"
         file = "/meas_MID00262_FID117439_SS_TRUFI_PSIR_SAX_FULL_SAMP_3CH/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      end
   end

elseif strncmp(id,"JBH",3)
   folder = "20230118_CS_LGE_JB/"; %#"20230125_CS_LGE_SKJ/"
   if FS==0
      if id=="JBH_3CH"
         file = "/meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      end
   else
      if id=="JBH_3CH"
         file = "/meas_MID00383_FID115210_SS_TRUFI_PSIR_3CH_FULL_SAMP/";% #meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      end
   end

elseif strncmp(id,"JGR",3)
   folder = "20230308_CS_LGE_JGR/" ;%#"20230125_CS_LGE_SKJ/"
   if FS==0
      if id=="JGR_3CH"
         file = "/meas_MID00097_FID134147_SS_TRUFI_CS_PSIR_3CH/"; %#meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      elseif id=="JGR_SAX_BASE"
         file = "/meas_MID00096_FID134146_SS_TRUFI_CS_PSIR_BASE_SAX/"; %#meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      end
   end

elseif strncmp(id,"GRH",3)
   folder = "20230118_CS_LGE_GR/" ;%#"20230125_CS_LGE_SKJ/"
   if FS==0
      if id=="GRH_3CH"
         file = "/meas_MID00488_FID115315_SS_TRUFI_CS_PSIR_3CH/"; %#meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      % elseif id=="JGR_SAX_BASE"
      %    file = "/meas_MID00096_FID134146_SS_TRUFI_CS_PSIR_BASE_SAX/"; %#meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      end
   end
elseif strncmp(id,"AT",2)
   folder = "20230816_CS_LGE_AT/" ;%#"20230125_CS_LGE_SKJ/"
   if FS==0
      if id=="AT_3CH"
         file = "/meas_MID00400_FID199583_SS_TRUFI_CS_PSIR_3_CH_SLICE_54_5/"; %#meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      % elseif id=="JGR_SAX_BASE"
      %    file = "/meas_MID00096_FID134146_SS_TRUFI_CS_PSIR_BASE_SAX/"; %#meas_MID00381_FID115208_SS_TRUFI_CS_PSIR_3CH
      end
   end

end


% load undersampled data 
path = "\\ak-isi01-sh2.prdnas1.osumc.edu\dhlri$\labs\CMRCT Lab Team\_ahmad_sultan\_shared_GPU_station\LGE Patient Datasets Preprocessing\data\preprocessed\";

file_r = file+"R"+num2str(R)+"\";

if FS==1
    path_l = path+folder+file_r;
else
    path_l = path+folder+file; 
end

set=0;

% load undersampled data 
formatSpec = '%.6f';
load(path_l+"yu"+"_R_"+num2str(R,formatSpec)+"_set_"+num2str(set)+".mat");
kdata = permute(yu, [3, 4, 1, 2]);
[nx,ny,nt,nc]=size(kdata);
class(kdata)
% check scale: 
min(abs(kdata(:))), max(abs(kdata(:)))

load(path+folder+file+"sen_cc_map"+".mat");

load(path_l+"ESPIRiT_crop_"+num2str(crop,formatSpec)+"_R_"+num2str(R, formatSpec)+"_set_"+num2str(set)+".mat");
% b1=rot90(permute(S, [2,3,1]), 2).*cc;
b1=rot90(permute(S, [2,3,1]).*cc, 2); % true


load(path_l+"th_sen_map_crop_"+num2str(crop,formatSpec)+'_R_'+num2str(R, formatSpec)+"_set_"+num2str(set)+".mat");
sen_msk = logical(th_msk);

% load mask
if FS==1
    load(path+folder+file+"xRef_N_"+num2str(nt)+"_set_"+num2str(set)+".mat")
    load(path+folder+file+"mask_PE"+num2str(ny)+"_FR"+num2str(nt)+"_R"+num2str(R)+".mat");
    mask=permute(repmat(samp, [1, 1, nx]), [3, 1, 2]);

else
    load(path+folder+file+"mask_R_"+num2str(R,formatSpec)+".mat");
    mask=permute(squeeze(samp(:,1,:,:)), [2, 3, 1]);
end
size(mask)
class(mask)


nt=N;
kdata = kdata(:,:,1:nt,:);
mask = mask(:,:,1:nt);
if FS==1
    mc = mc(1:nt,:,:);
end


% L+S reconstruction ******************************************************

% for GR and JGR 3CH:
% mask0=zeros(nx,ny+1,nt); mask0(:,1:end-1,:)=mask;
% b10=zeros(nx,ny+1,nc); b10(:,1:end-1,:)=b1;
% kdata0=zeros(nx,ny+1,nt,nc); kdata0(:,1:end-1,:,:)=kdata;

param.E=Emat_xyt(mask,b1); % encoding matrix (forward operator): input: mask and sen maps
param.d=kdata; % sacle=1; given undersampled kspace measurement
param.T=TempFFT(3); % sparsifying transform: Temp FFT here

param.nite=200; % 50 % num iterations
param.tol=0.0001; % optimization stopping criteria: when update is too small


%% training parameters: 
sv=1;
param.lambda_L= 0.0025;%0.0035; % strength of Low-Rank % default 0.005
param.lambda_S=  0.03;%0.04; % strength of Sparsity % default 0.01

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
end

clip=0.5;
figure;
subplot(3,1,1),imagesc(abs(LplusSd),[0,clip*max(abs(LplusSd(:)))]); axis off; axis image; colormap(gray);title('L+S');
subplot(3,1,2),imagesc(abs(Ld),[0,clip*max(abs(Ld(:)))]); axis off; axis image; colormap(gray);title('L');
subplot(3,1,3),imagesc(abs(Sd),[0,clip*max(abs(Sd(:)))]); axis off; axis image; colormap(gray);title('S');


% results
LplusS = LplusS.*sen_msk;
xLS = permute(LplusS, [3, 1, 2]);
% shift adjust:
xLS = correct_shift_LplusS(xLS);
if sv==1
save(path_l+'LplusS'+'_N_'+num2str(nt)+'_R_'+num2str(R, formatSpec)+'.mat', 'xLS');end

if FS==1

mc = permute(permute(mc, [2,3,1]).*sen_msk, [3,1,2]); 

xAbs = abs(mc); % (32, 160, 96)
% x = complex_to_real_plus_imag(mc);

xHat = complex_to_real_plus_imag(xLS);
xHatAbs = abs(xLS);

% % FAST :
nmse= mean(abs(mc - xLS).^2, [2,3]) ./ mean(abs(mc).^2, [2,3]);
nmse_mean = 10*log10(mean(nmse))

% NMSE ROI:
ROI_offset = floor(0.2*nx);
nmse_ROI= mean(abs(mc(:,ROI_offset+1:end-ROI_offset,:) - xLS(:,ROI_offset+1:end-ROI_offset,:)).^2, [2,3]) ./ mean(abs(mc(:,ROI_offset+1:end-ROI_offset,:)).^2, [2,3]);
nmse_ROI_mean = 10*log10(mean(nmse_ROI))

% % error maps:
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



function y = correct_shift_LplusS(x)
y0=zeros(size(x)); y0(:,:,1)=x(:,:,end); y0(:,:,2:end)=x(:,:,1:end-1);
y=zeros(size(x)); y(:,1,:)=y0(:,end,:); y(:,2:end,:)=y0(:,1:end-1,:);
end
% % 
% figure
% imagesc(abs(permute(squeeze(samp(:,1,:,1)), [2,1])));

% nmse_new = zeros(1,nt);
% for i=1:nt
% nmse_ = mean(abs(mc - xLS).^2, [2,3]) ./ mean(abs(mc).^2, [2,3]);
%     nmse(1,i) = mean((x(i*2-1:i*2,:,:) - xHat(i*2-1:i*2,:,:)).^2, 'all') / mean(x(i*2-1:i*2,:,:).^2,'all'); 
%     10*log10(nmse(1,i))
%     ssim(1,i) = 

% end