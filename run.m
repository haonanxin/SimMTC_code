clear;clc;close all;
addpath('funs');
addpath('data');

dataset_name='NGs';
load([dataset_name,'.mat'])
v = length(X);
k = length(unique(Y));
n = length(Y);

%% Normalization
for i = 1 :v
    for  j = 1:n
        X{i}(j,:) = (X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
    end
end

%% Parameter Setting of Digit4k without normalization        ACC = 0.98
% lambda=10.^[-1]; 
% anchorRate=1;
% mu=10^(-5);

%% Parameter Setting of LandUse_21 without normalization     ACC = 0.74
% lambda=10.^[-1]; 
% anchorRate= 1;
% mu=10^(-5);

%% Parameter Setting of MNIST_mv without normalization       ACC = 0.99
% lambda=10.^[-2]; 
% anchorRate=0.8;
% mu=10^(-5);

%% Parameter Setting of NGs with normalization               ACC = 1
lambda=10.^[-2]; 
anchorRate=0.3;
mu=10^(-5);

%% Optimization of SimMTC
anchorNum = fix(n * anchorRate);
opt1.style = 1;
opt1.IterMax = 50;
opt1.toy = 0;
[~, B_init] = FastmultiCLR(X, k, anchorNum, opt1, 5);

[F,obj] = SimMTC_obj(B_init,k,lambda,1,mu);
% [F,obj] = SHGT_tensor_simple_eig2_obj(B_init,k,lambda,1,mu);
Y_pre=kmeans(F,k,'Replicates',100,'MaxIter',50);
my_result = ClusteringMeasure_new(Y, Y_pre);


disp(['********************************************']);
disp(['Running SimMTC on ',dataset_name,' to obtain ACC: ', num2str(my_result.ACC)]);
disp(['********************************************']);



