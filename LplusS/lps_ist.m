function [L,S] = lps_ist(param)
%
% L+S reconstruction of undersampled dynamic MRI data using iterative
% soft-thresholding of singular values of L and entries of TS
%
% [L,S]=lps_ist(param)
%
% Input variables and reconstruction parameters must be stored in the
% struct param
%
% param.d: undersampled k-t data (nx,ny,nt,nc)
% param.E: data acquisition operator
% param.T: sparsifying transform
% param.lambda_L: nuclear-norm weight
% param.lambda_S: l1-norm weight
%
% Ricardo Otazo (2013)



M=param.E'*param.d;
clip=0.3;
figure;imagesc(abs(M(:,:,1)),[0,clip*max(abs(M(:)))]); axis off; axis image; colormap(gray);ylabel('L+S'); title("init. first frame before training")
% fprintf(M(:,:,1));
%M(:,:,1)

[nx,ny,nt]=size(M);
M=reshape(M,[nx*ny,nt]); % each col a frame
Lpre=M;S=zeros(nx*ny,nt); 
ite=0;

% % load ref:
% formatSpec = '%.6f';
% nIter=250;
% tau=1e-6;
% 
% path = "R:\CMRCT Lab Team\_ahmad_sultan\_shared_GPU_station\MRXCAT_xuan\data\digit_patients_with_without_lesion\without_lesion\female_pt71\";
% load(path+"xRef_N_"+num2str(nt)+"_ADMMnIters_"+num2str(nIter)+"_tau_"+num2str(tau,formatSpec)+".mat");

% path = "R:\CMRCT Lab Team\_ahmad_sultan\_shared_GPU_station\LGE Patient Datasets Preprocessing\data\preprocessed\";
% folder="20230308_CS_LGE_JGR\new sen cc maps\"; %20230125_CS_LGE_SKJ %20230118_CS_LGE_JB
% file="meas_MID00099_FID134149_SS_TRUFI_PSIR_3CH_FULL_SAMP\";%"meas_MID00490_FID115317_SS_TRUFI_PSIR_3CH_FULL_SAMP\"; 
% set=0; nt=32;
% load(path+folder+file+"xRef_N_"+num2str(nt)+"_set_"+num2str(set)+".mat");



% fprintf('\n ********** L+S reconstruction **********\n')
% iterations
while(1)
	ite=ite+1;
	% low-rank update
	M0=M;
	[Ut,St,Vt]=svd(M-S,0);
	St=diag(SoftThresh(diag(St),St(1)*param.lambda_L));
	L=Ut*St*Vt'; % just the transpose for inverse


	% sparse update
	S=reshape(param.T'*(SoftThresh(param.T*reshape(M-Lpre,[nx,ny,nt]),param.lambda_S)),[nx*ny,nt]);

	% data consistency
	resk=param.E*reshape(L+S,[nx,ny,nt])-param.d; % [nx, ny, nt, nc]
	M=L+S-reshape(param.E'*resk,[nx*ny,nt]); % [nx*ny,nt]

    %% plot recon first frame each iter.
%     frames = reshape(M, [nx,ny,nt]);
%     frame = frames(:,:,1);
%     if ite==1
%         fprintf("L(St) Max. %f3, S Max. %f3\n", max(abs(L(:))), max(abs(S(:))));
%     end

	% L_{k-1} for the next iteration
	Lpre=L;


% 	% print cost function and solution update
% 	tmp2=param.T*reshape(S,[nx,ny,nt]);
%     % indiv. losses:
%     dc_loss = norm(resk(:),2)^2;
%     lr_loss = param.lambda_L*sum(diag(St));
%     s_loss = param.lambda_S*norm(tmp2(:),1);
% % 
%     loss = dc_loss+lr_loss+s_loss; % Loss
% %     fprintf(' indiv. losses: \nDC loss: %f3, LR loss: %f3, S loss: %f3\n', dc_loss, lr_loss, s_loss); 

%     % calculate NMSE 
%     L_=reshape(L,nx,ny,nt);
%     S_=reshape(S,nx,ny,nt);
%     L__ = rot90(L_, 2);
%     S__ = rot90(S_, 2);
%     LplusS_=L__+S__; 
%     xLS_ = permute(LplusS_, [3, 1, 2]);
%     xLS__ = correct_shift_LplusS(xLS_);
%     mc = permute(permute(mc, [2,3,1]), [3,1,2]); 
%     nmse= mean(abs(mc - xLS__).^2, [2,3]) ./ mean(abs(mc).^2, [2,3]);
%     nmse_mean = 10*log10(mean(nmse));
% 
% 	fprintf(' ite: %d , update: %f3, mean nmse: %f3\n', ite,norm(M(:)-M0(:))/norm(M0(:)), nmse_mean); 

	fprintf(' ite: %d \n', ite); 


% 	% stopping criteria 
	if (ite > param.nite) || (norm(M(:)-M0(:))<param.tol*norm(M0(:))), break;end
end
L=reshape(L,nx,ny,nt);
S=reshape(S,nx,ny,nt);
end

% soft-thresholding function
function y=SoftThresh(x,p)
y=(abs(x)-p).*x./abs(x).*(abs(x)>p);
y(isnan(y))=0;
end    

function y = correct_shift_LplusS(x)
y0=zeros(size(x)); y0(:,:,1)=x(:,:,end); y0(:,:,2:end)=x(:,:,1:end-1);
y=zeros(size(x)); y(:,1,:)=y0(:,end,:); y(:,2:end,:)=y0(:,1:end-1,:);
end