%% Load Training Dataset
clc;clear;close all;restoredefaultpath;

addpath('Models')
addpath('Data')
load 'Normalised Tip Contact Test'

h = 0.1;
L = size(curvature,2); % no. of time step % no. of state
tn = linspace(1,L,L);
tv = linspace(0,L*h,L);

% figure
% plot(tv,curvature)
% figure
% plot(tv,force)
%% Simulate Sensor Fault/Noise
flex_clean = lowpass(flex,0.1,1/h);
%snr(flex_clean,flex - flex_clean)

% Fault
slp1 = -0.002*((max(flex)-min(flex))/2);
Fault1 = slp1*(tv - tv(1));
flex_fault = max(flex + Fault1, -1.3945);

%sensor = flex;
%sensor = flex_noisy;
sensor = flex_fault;

%figure
plot(tv,sensor,tv,flex)
%% UI-UKF (non-RNN)
% Initiation
NH = 32;                % no. of hidden states
ns = NH;                % no. of state
nm = size(flex,1);      % no. of measurement
NS = 2*ns + 1;          % no. of sigma points

x_h = zeros(ns,L+1);
xh = zeros(ns,L);

hu = zeros(NH,L);

Ds = 0;
Es = 0;
D = zeros(nm,L);
E = zeros(nm,L);
Q = zeros(ns,ns,L);
R = zeros(nm,L);

% UI-Estimator Initiation
nu = size(force,1);        % no. of input
uh = zeros(nu,L);

% Initial State
x_h(:,2) = x_init;

model_number = 0;

% Noise Covariances
P_ = 10*diag(ones(ns,1));
We = 0.02*ones(ns,1);

% Adaptive parameters
mv = 0;
Beta = 0.99;
Lm = L;
Alpha = 1;

% Model 1: P_initial = 5 Adaptive Filtering = 0.02 Sensor Degradation = 0.02 Beta = 0.99

Ve = 0*ones(nm,1);
Q(:,:,2) = We*We';
R(2) = Ve*Ve';

% UT Initiation
alpha = 1;
ki = 0;
beta = 2;
lambda = alpha^2*(ns + ki) - ns;
c = ns + lambda;
Wm = [lambda/c , 0.5/c+zeros(1,2*ns)];
Wc = Wm;
Wc(1) = Wc(1) + (1 - alpha^2 + beta);
c = sqrt(c);

% Generate Sigma Points for Initial Correction
clear XD XH
XH = sigmas(x_h(:,2),P_,c);
XD = XH - x_h(:,2);

beep on
tic;
for k = 2:L
    
    % Measurement UT Transformation
    input = pressure(k-1);
    [yh_,Pyy,~,YD,~] = HUT(model_number,XH,input,Wm,Wc,R(k));
    
    % Correct
    Pxy = XD*diag(Wc)*YD';
    K = Pxy*inv(Pyy);
    P = P_ - K*Pxy';
    
    d = sensor(k) - yh_;
    D(k) = d*d';
    
    xh(:,k) = x_h(:,k) + K*d;
    
    % Generate Sigma Points
    XH = sigmas(xh(:,k),P,c);
    
    % Calculate UT transformed posterior measurement covariance
    input = pressure(k-1);
    [yh,~,~,~,PT] = HUT(model_number,XH,input,Wm,Wc,R(k));
    
    eps = sensor(k) - yh;
    E(k) = eps*eps';
    
    % Moving Average
    if k < Lm
        Ds = sum(D)/length(D);
        Es = sum(D)/length(D);
    else
        Ds = sum(D(k-Lm+1:k))/Lm;
        Es = sum(E(k-Lm+1:k))/Lm;
    end
    
    % Optimization for UI Estimation
    state = model(model_number,'hidden2state',xh(:,k));
    input = [pressure(k-1);state];
    [uh(k),hu(:,k+1)] = model(model_number,'forceEst_RNN',input,hu(:,k));
    uh(k) = max(min(uh(k),3.613),-0.7324);
    %uh(k) = force(k);
    
    % Predict
    
    % Adaptive KF covariances
    if mv == 1
    Q(:,:,k) = Alpha*Q(:,:,2) + (1-Alpha)*(K*Ds*K');
    R(k+1) = Beta*R(2) + (1-Beta)*(Es + PT);
    else
    Q(:,:,k) = Alpha*Q(:,:,2) + (1-Alpha)*(K*D(k)*K');
    R(k+1) = Beta*R(k) + (1-Beta)*(E(k) + PT);
    end
    
    % System UT Transformation
    input = [pressure(k);uh(k)];
    [x_h(:,k+1),P_,XH,XD,~] = FUT(model_number,XH,input,Wm,Wc,Q(:,:,k));
    
    % Progress
    %k
    
end
run_time = toc;
% beep
beep off

% Check EKF result
xh = model(model_number,"hidden2state",xh);
rms_xh = nrmse(curvature,xh,range_c)
xhf = lowpass(xh,0.1,1/h);
snr_xh = snr(xhf,xh - xhf)
figure
plot(tv,curvature,'b',tv,xh,'r')
legend('Actual','Estimated');
title('State (Curvature) Estimation')

% Check UI-Estimator result
rms_uh = nrmse(force,uh,range_f)
uhf = lowpass(uh,0.1,1/h);
snr_uh = snr(uhf,uh - uhf)
figure
plot(tv,force','b',tv,uh,'r')
legend('Actual','Estimated');
title('Input (Force) Estimation')

% Check Noise Covariances
% figure
% plot(tv,R(1:end-1))
% title('R')
 
% % Check innovation and residual
% figure
% subplot(2,1,1)
% plot(tv,D)
% title('Innovation')
% subplot(2,1,2)
% plot(tv,E)
% title('Residual')
%% Direct LSTM
NH = 32;                  % no. of hidden states
hd = 0.01*ones(NH,L);
Lf = 1;
w1 = flip([1:Lf]');
w2 = sum(w1);

for k = 1:L
[output(:,k),hd(:,k+1)] = model(model_number,'direct_RNN', [pressure(k);sensor(k)], hd(:,k));

 if k < Lf
        output(:,k) = sum(output,2)/length(output);
    else
        output(:,k) = sum(output(:,k-Lf+1:k),2)/Lf;
        %output(:,k) = output(:,k-Lf+1:k)*w1/w2;
    end
end

xd = output(2,:);
ud = output(1,:);

rms_xd = nrmse(curvature,xd,range_c)
xdf = lowpass(xd,0.1,1/h);
snr_xd = snr(xdf,xd - xdf)
figure
plot(tv,curvature,'b',tv,xd,'r')
% axis([0,ts(end),x_min,x_max])
legend('Actual','Estimated');
title('State (Curvature) Estimation')

% Check UI-Estimator result
rms_ud = nrmse(force,ud,range_f)
udf = lowpass(ud,0.1,1/h);
snr_ud = snr(udf,ud - udf)
figure
plot(tv,force','b',tv,ud,'r')
% axis([0,tv(end),u_min,u_max])
legend('Actual','Estimated');
title('Input (Force) Estimation')
%% Direct LSTM - MAF
NH = 32;                  % no. of hidden states
hd = 0.01*ones(NH,L);
Lf = 12;
w1 = flip([1:Lf]');
w2 = sum(w1);

for k = 1:L
[output(:,k),hd(:,k+1)] = model(model_number,'direct_RNN', [pressure(k);sensor(k)], hd(:,k));

 if k < Lf
        output(:,k) = sum(output,2)/length(output);
    else
        output(:,k) = sum(output(:,k-Lf+1:k),2)/Lf;
        %output(:,k) = output(:,k-Lf+1:k)*w1/w2;
    end
end

xm = output(2,:);
um = output(1,:);

rms_xm = nrmse(curvature,xm,range_c)
xmf = lowpass(xm,0.1,1/h);
snr_xm = snr(xmf,xm - xmf)
figure
plot(tv,curvature,'b',tv,xm,'r')
% axis([0,ts(end),x_min,x_max])
legend('Actual','Estimated');
title('State (Curvature) Estimation')

% Check UI-Estimator result
rms_um = nrmse(force,um,range_f)
umf = lowpass(um,0.1,1/h);
snr_um = snr(umf,um - umf)
figure
plot(tv,force','b',tv,um,'r')
% axis([0,tv(end),u_min,u_max])
legend('Actual','Estimated');
title('Input (Force) Estimation')
%% Data Partition
close all;

partition = 1;
for i = 2:length(dutycycle)  
    if dutycycle(i) ~= dutycycle(i-1)
        partition = [partition,i];
    end
end
partition = [partition,length(dutycycle)];
%% Statistical Test
for i = 1:length(partition)-1
    
    if i == length(partition)-1
        sample = partition(i):partition(i+1);
    else
        sample = partition(i):partition(i+1)-1;
    end
    
    RMS_xh(i) = nrmse(curvature(sample),xh(sample),range_c);
    RMS_xd(i) = nrmse(curvature(sample),xd(sample),range_c);
    RMS_xm(i) = nrmse(curvature(sample),xm(sample),range_c);
    RMS_uh(i) = nrmse(force(sample),uh(sample),range_f);
    RMS_ud(i) = nrmse(force(sample),ud(sample),range_f);
    RMS_um(i) = nrmse(force(sample),um(sample),range_f);
    SNR_xh(i) = snr(xhf(sample), xh(sample) - xhf(sample));
    SNR_xd(i) = snr(xdf(sample), xd(sample) - xdf(sample));
    SNR_xm(i) = snr(xmf(sample), xm(sample) - xmf(sample));
    SNR_uh(i) = snr(uhf(sample), uh(sample) - uhf(sample));
    SNR_ud(i) = snr(udf(sample), ud(sample) - udf(sample));
    SNR_um(i) = snr(umf(sample), um(sample) - umf(sample));
end

MEAN(1,1) = mean(RMS_xh);
MEAN(2,1) = mean(RMS_xd);
MEAN(3,1) = mean(RMS_xm);
MEAN(1,2) = mean(SNR_xh);
MEAN(2,2) = mean(SNR_xd);
MEAN(3,2) = mean(SNR_xm);
MEAN(1,3) = mean(RMS_uh);
MEAN(2,3) = mean(RMS_ud);
MEAN(3,3) = mean(RMS_um);
MEAN(1,4) = mean(SNR_uh);
MEAN(2,4) = mean(SNR_ud);
MEAN(3,4) = mean(SNR_um);

STD(1,1) = std(RMS_xh);
STD(2,1) = std(RMS_xd);
STD(3,1) = std(RMS_xm);
STD(1,2) = std(SNR_xh);
STD(2,2) = std(SNR_xd);
STD(3,2) = std(SNR_xm);
STD(1,3) = std(RMS_uh);
STD(2,3) = std(RMS_ud);
STD(3,3) = std(RMS_um);
STD(1,4) = std(SNR_uh);
STD(2,4) = std(SNR_ud);
STD(3,4) = std(SNR_um);

[hxe,pxe,cxe] = ttest2(RMS_xh,RMS_xd,'Alpha',0.05,'Tail','right','Vartype','unequal');
[hue,pue,cue] = ttest2(RMS_uh,RMS_ud,'Alpha',0.05,'Tail','left','Vartype','unequal');
[hxn,pxn,cxn] = ttest2(SNR_xh,SNR_xd,'Alpha',0.05,'Tail','right','Vartype','unequal')
[hun,pun,cun] = ttest2(SNR_uh,SNR_ud,'Alpha',0.05,'Tail','right','Vartype','unequal')