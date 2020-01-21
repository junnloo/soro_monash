function [y,P,Y,Yd,PT] = HUT(model_number,hidden,input,Wm,Wc,R)
%Unscented Transformation
%Input:
%        f: nonlinear map
%        X: sigma points
%       Wm: weights for mean
%       Wc: weights for covraiance
%        n: numer of outputs of f
%        R: additive covariance
%Output:
%        y: transformed mean
%        Y: transformed smapling points
%        P: transformed covariance
%       Y1: transformed deviations

state = model(model_number,"hidden2state",hidden);
input = [repmat(input,size(state));state];
Y = model(model_number,"measurement_FC",input);
y = Y*Wm';
Yd = Y - y;
PT = Yd*diag(Wc)*Yd';
P = PT + R;