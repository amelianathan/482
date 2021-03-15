% Amelia Nathan
% AMATH 482, Winter 2021, Jason Bramburger
% Homework 1

%% Set up, initialization
% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49
% (space by time) matrix called subdata 5

L = 10; % spatial domain
n = 64; % arbitrary value; chosen as power of 2 for efficient execution
x2 = linspace(-L,L,n+1); % Create a vector with 65 equally spaced sections
x = x2(1:n); 
y =x; 
z = x; % initializing the spatial domain for graph 
% Rescale frequency domain to fit our spatial domain
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 

ks = fftshift(k); % shift frequencies back to center of spectrum
[X,Y,Z]=meshgrid(x,y,z); 

[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% create initial figure plotting noisy data
for j = 1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
    isosurface(X,Y,Z,abs(Un)/M, 0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    title('Original Submarine Data', 'Fontsize', [26])
    xlabel('x position', 'Fontsize', [20])
    ylabel('y position', 'Fontsize', [20]) 
    zlabel('z position','Fontsize', [20])
end
%% Algorithm 1 - Averaging

ave = zeros(1,n); % initializes average

for j = 1:49 % Loop through for each realization in the 24 hour period
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    Unt = fftn(Un); % to frequency domain
    ave = Unt + ave; % summing frequency realization
end 
ave = abs(fftshift(ave))/49; 
[center_signal, index] = max(ave(:)); % find maximum signal 
% so we know where to center it (at the index) 
[xind, yind, zind] = ind2sub(size(ave), index);
x0 = Kx(xind, yind, zind); % represents centered frequency - 
% this is the submarine's signature (x0, y0, z0)
y0 = Ky(xind, yind, zind); 
z0 = Kz(xind, yind, zind);

%% Algorithm 2 - Filter Function
xpos = zeros(0,49);
ypos = zeros(0,49);
zpos = zeros(0,49);

tau = 0.2; % assign as window we are keeping to analyze
% Defining filter
filter = exp(-tau*(Kx - x0).^2).*exp(-tau*(Ky - y0).^2).*exp(-tau*(Kz - z0).^2); 
filter = fftshift(filter);
% Apply filter at each point in time
for j = 1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    Unt = fftn(Un);
    unft = filter.*Unt; % Applying filter to signal
    % inverse fourier transform to get filtered data in space and time
    unf = ifftn(unft); 
    [maxintime, index2] = max(unf(:));
    [xind2, yind2, zind2] = ind2sub(size(unf), index2); 
    xpos(j) = X(xind2, yind2, zind2);
    ypos(j) = Y(xind2, yind2, zind2);
    zpos(j) = Z(xind2, yind2, zind2);
end 
% plot submarine path based on location coordinates
plot3(xpos, ypos, zpos, 'r', 'Linewidth', 2) 
axis([-20 20 -20 20 -20 20]), grid on, drawnow
title('Submarine Path', 'Fontsize', [26])
xlabel('x position', 'Fontsize', [20])
ylabel('y position', 'Fontsize', [20]) 
zlabel('z position','Fontsize', [20])

xycoordinates = [xpos; ypos]; % for table generation



