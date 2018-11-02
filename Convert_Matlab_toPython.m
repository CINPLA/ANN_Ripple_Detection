clear all; close all;clc

%%
data = load('data/lfpTrace_ripples.mat');

X = data.lfp;
Y = data.rippleLocs;

%%
figure
plot(X)