% AMATH 482
% Assignment 5 - Video Foreground and Background Isolation


movie_name = 'monte_carlo_low.mp4';

vid = VideoReader(movie_name);
numFrames = vid.NumFrames;
for i = 1:numFrames
    colored_frame = read(vid,i);
    % imshow(colored_frame); 
    x = rgb2gray(colored_frame);
    x = imadjust(x);
    x_reshaped = reshape(x,[],1);
    data(:,i) = x_reshaped;
end
data = double(data);
data1 = data(:,1:end-1);
data2 = data(:,2:end);

[U, Sigma, V] = svd(data1, 'econ');


lambda = diag(Sigma).^2; % vector of singular values

subplot(1,3,1)
plot(1:length(lambda), lambda/sum(lambda)*100, '.','MarkerFaceColor','#0072BD', 'MarkerSize', 20); 
title('Singular Value Energies','Fontsize', 16);
xlabel('ith singular value')
ylabel('percent of total energy')
subplot(1,3,2)
plot(1:10, lambda(1:10,:)/sum(lambda(1:10))*100, '.','MarkerFaceColor', '#0072BD', 'MarkerSize', 20);
title('Singular Value Energies','Fontsize', 16);
xlabel('ith singular value')
ylabel('percent of total energy')


rank = 5;
U = U(:,1:rank);
Sigma = Sigma(1:rank,1:rank);
V = V(:, 1:rank);



dt = 1;
t = 1:numFrames-1;
S = U'*data2*V*diag(1./diag(Sigma));
[ev, D] = eig(S);
mu = diag(D);
omega = log(mu)/(dt); % the omegas close to 0 represent dmd modes that best describe background 

subplot(1,3,3)
plot(omega, '.', 'MarkerSize', 20)
title('Eigenvalues','Fontsize', 16)
omega = omega.*(abs(omega)<0.05);
Phi = U*ev;
Phi = Phi(:,abs(omega)<0.05);
y0 = Phi\data1(:,1); % pseudoinverse
y0 = y0.*(abs(omega)<0.05);
sgtitle('Monte Carlo', 'Fontsize', 22)
u_modes = zeros(length(y0),length(t));
for j = 1:length(t)
    u_modes(:,j) = y0.*exp(omega*t(j));
end
u_dmd = Phi*u_modes;
%% For cars
x_lowrank = (abs(u_dmd));
x_sparse = data1 - x_lowrank;

r = x_sparse.*(x_sparse < 0);
x_sparse = uint8(x_sparse - r); 
% x_sparse = uint8(x_sparse.*(x_sparse>700) - r).*3 - uint8(50);
x_lowrank_fr = uint8(x_lowrank + r);
reconstruction_car = uint8(x_sparse + x_lowrank_fr);
x_lowrank = uint8(x_lowrank); % assignment says add r, but in practice this adds the car back in
figure(7)
subplot(2,2,1)
imshow(reshape(x_lowrank(:,100),[],vid.Width)) % background
title('Background', 'Fontsize', 16)
subplot(2,2,2)
imshow(reshape(x_sparse(:,100),[],vid.Width))
title('Foreground Method 1', 'Fontsize', 16)

x_lowrank = (abs(u_dmd));
x_sparse = data1 - x_lowrank;
r = x_sparse.*(x_sparse < 0);
x_sparse = uint8(x_sparse.*(x_sparse>700) - r).*3 - uint8(50); % second method
subplot(2,2,3)
imshow(reshape(x_sparse(:,100),[],vid.Width))
title('Foreground Method 2', 'Fontsize', 16)
subplot(2,2,4)
imshow(reshape(reconstruction_car(:,100),[],vid.Width))
title('Reconstructed', 'Fontsize', 16)
%%
figure(2)
for frame = 1:numFrames - 1 % foreground
    imshow(reshape(x_sparse(:,frame),[],vid.Width))
end

% plot reconstruction

%% For Skier 

x_lowrank = (abs(u_dmd));
x_sparse = data1 - x_lowrank;
figure(5)
r = x_sparse.*(x_sparse < 0);
%subplot(2,2,2)
x_sparse = uint8(x_sparse - r); % original, but cannot see skier as they blend into the black
imshow(reshape(x_sparse(:,300),[],vid.Width))
title('Foreground Method 1', 'Fontsize', 16)
% x_lowrank = uint8(x_lowrank); % assignment says add r
x_lowrank = uint8(x_lowrank + r);
reconstruction = x_sparse + x_lowrank;
imshow(reshape(reconstruction(:,300),[],vid.Width))
title('Reconstructed Image', 'Fontsize', 20)
%%

subplot(2,2,1)
imshow(reshape(x_lowrank(:,300),[],vid.Width)) % background
title('Background', 'Fontsize', 16)

x_lowrank = (abs(u_dmd));
x_sparse = data1 - x_lowrank;

r = x_sparse.*(x_sparse < 0);

x_sparse = uint8(x_sparse.*(x_sparse>500) - r).*5 - uint8(50); % method 2 - magnifies the skier
subplot(2,2,3)
imshow(reshape(x_sparse(:,300),[], vid.Width))
title('Foreground Method 2', 'Fontsize', 16)

% foreground method 3
x_lowrank = (abs(u_dmd));
x_sparse = data1 - x_lowrank;

r = x_sparse.*(x_sparse < 0);
x_sparse = (x_sparse - min(x_sparse(:,:)))./(max(x_sparse(:,:)) - min(x_sparse(:,:)));
subplot(2,2,4)
imshow(reshape(x_sparse(:,300),[], vid.Width))
title('Foreground Method 3', 'Fontsize', 16)

%%
figure(2)
for frame = 1:numFrames-1
    imshow(reshape(x_sparse(:,frame),[], vid.Width)) %foreground
end
%%
figure(6)
subplot(1,3,1)
imshow(reshape(x_sparse(:,150),[], vid.Width))
title('Frame 150', 'Fontsize', 16)
subplot(1,3,2)
imshow(reshape(x_sparse(:,225),[], vid.Width))
title('Frame 225', 'Fontsize', 16)
subplot(1,3,3)
imshow(reshape(x_sparse(:,300),[], vid.Width))
title('Frame 300', 'Fontsize', 16)
sgtitle('Ski Drop Method 3', 'Fontsize', 20)