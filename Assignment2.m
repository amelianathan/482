% Amelia Nathan
% AMATH 482, Assignment 2

figure(1)
[y, Fs] = audioread('GNR.m4a'); % Guns N's Roses Sweet Child O' Mind
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O Mine');
p8 = audioplayer(y,Fs); % playblocking(p8);

figure(2)
[y2, Fs2] = audioread('Floyd.m4a'); % Pink Floyd Comfortably Numb
tr_floyd = length(y2)/Fs2; % record time in seconds
plot((1:length(y2))/Fs2,y2);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb');
p8 = audioplayer(y2,Fs2); 
% playblocking(p8);

%% Sweet Child O' Mine, Guns N Roses, Guitar 
L = tr_gnr; % How long the song is
s = y.';
n = length(s); % arbitrary value; 
% chosen as power of 2 for efficient execution; number of data points
x2 = linspace(-L,L,n+1); % Create a vector with equally spaced sections
t = x2(1:n); 
% Rescale frequency domain to fit our spatial domain
k = (1/L)*[0:(n/2 - 1) -n/2:-1]; % Changing this to hertz
ks = fftshift(k); % shift frequencies back to center of spectrum


% Sliding window across domain

a = 300;
tau = 0:0.05:14; % can make this smaller

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*s;
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(3)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0, 1000],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
yline(220, 'w', '220: A')
yline(277, 'w', '277.183: C#')
yline(369, 'w', '369.99: F#')
yline(415, 'w', '415.305: G#')
yline(554, 'w', '554.36: C#')
yline(698, 'w', '698.456: F')
yline(739, 'w', '739.989: F#')
title('Guitar in Sweet Child O Mine')

%% Comfortably Numb, Pink Floyd Pt. 1
L = tr_floyd*0.5; % How long the song is; using two half song increments
s = y2.';
s = s(1:1317960); % raw signal
n = length(s); 
x2 = linspace(-L,L,n+1); % Create a vector with equally spaced sections
t = x2(1:n); 
% Rescale frequency domain to fit our spatial domain
k = (1/L)*[0:(n/2 - 1) -n/2:-1]; % Changing this to hertz
ks = fftshift(k); % shift frequencies back to center of spectrum


% Sliding window across domain

a = 300;
tau = 0:0.25:30;

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*s;
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(4)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0, 1000],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
title('Floyd, Comfortably Numb, First Half')
yline(123, 'w', '123.47: B')
yline(220, 'w', '220: A')
yline(277, 'w', '277.183: C#')
yline(369, 'w', '369.99: F#')
yline(415, 'w', '415.305: G#')
yline(554, 'w', '554.36: C#')
yline(698, 'w', '698.456: F')
yline(739, 'w', '739.989: F#')
%% Comfortably Numb, Pink Floyd Pt. 2 

L = tr_floyd*0.5; % How long the song is; using two half song increments
s = y2.';
s = s(1317961:2635920); % s is raw signal; using second half of song here
n = length(s); % arbitrary value; 
% chosen as power of 2 for efficient execution; number of data points
x2 = linspace(-L,L,n+1); % Create a vector with 65 equally spaced sections
t = x2(1:n); 
% Rescale frequency domain to fit our spatial domain
k = (1/L)*[0:(n/2 - 1) -n/2:-1]; % Changing this to hertz
ks = fftshift(k); % shift frequencies back to center of spectrum


% Sliding window across domain

a = 200;
tau = 0:0.25:30;

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*s;
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(5)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0, 1000],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
title('Floyd, Comfortably Numb, Second Half')

xticklabels({'30', '35', '40', '45', '50', '55','60'})
yline(123, 'w', '123.47: B')
yline(220, 'w', '220: A')
yline(277, 'w', '277.183: C#')
yline(369, 'w', '369.99: F#')
yline(415, 'w', '415.305: G#')
yline(554, 'w', '554.36: C#')
yline(698, 'w', '698.456: F')
yline(739, 'w', '739.989: F#')

%% Part 2 - ISOLATING BASS - pt.1
% Run this section for both halves of the song

% Apply filter in frequency space
floyd_fft = fft(s);
shannon = ones(1, length(s)).*(abs(ks) < 200);
floyd_fft = floyd_fft.*fftshift(shannon);
floyd_fft = ifft(floyd_fft); 


a = 250;
tau = 0:0.25:30;

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*floyd_fft;
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(6)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0, 250],'Fontsize',16) 
% Adjusted y limits to center and focus on bass
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
yline(87, 'w', '87.307: F', 'LineWidth', 1)
yline(98, 'w', '97.999: G')
yline(110, 'w', '110.00: A')
yline(123, 'w', '123.47: B')
title('Comfortably Numb Bass, First Half')
%% Part 2 - ISOLATING BASS - pt.2

% Apply filter in frequency space
floyd_fft = fft(s);
shannon = ones(1, length(s)).*(abs(ks) < 200);
floyd_fft = floyd_fft.*fftshift(shannon);
floyd_fft = ifft(floyd_fft); 


a = 250;
tau = 0:0.25:30;

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*floyd_fft;
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(6)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0, 250],'Fontsize',16) 
% Adjusted y limits to center and focus on bass
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
xticklabels({'30', '35', '40', '45', '50', '55','60'})
yline(87, 'w', '87.307: F')
yline(98, 'w', '97.999: G')
yline(110, 'w', '110.00: A')
yline(123, 'w', '123.47: B')
title('Comfortably Numb Bass, Second Half')
%% ISOLATING GUITAR - pt.1

% Apply filter in frequency space
floyd_fft = fft(s);
shannon = ones(1, length(s)).*(abs(ks-350) < 150);
floyd_fft = floyd_fft.*fftshift(shannon);
floyd_fft = ifft(floyd_fft); 


a = 200;
tau = 0:0.2:30;

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*floyd_fft;
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(7)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[200, 500],'Fontsize',16) 
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
yline(246, 'w', '246.94: B')
yline(277, 'w', '277.18: C#')
yline(369, 'w', '369.99: F#')
yline(415, 'w', '415.305: G#')
title('Comfortably Numb Guitar, First Half')
%% ISOLATING GUITAR - pt.2

% Apply filter in frequency space
floyd_fft = fft(s);
shannon = ones(1, length(s)).*(abs(ks-350) < 150);
floyd_fft = floyd_fft.*fftshift(shannon);
floyd_fft = ifft(floyd_fft); 


a = 200;
tau = 0:0.2:30;

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*floyd_fft;
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); % We don't want to scale it
end

figure(7)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[200, 500],'Fontsize',16) 
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')
xticklabels({'30', '35', '40', '45', '50', '55','60'})
yline(220, 'w', '220: A')
yline(277, 'w', '277.183: C#')
yline(369, 'w', '369.99: F#')
yline(415, 'w', '415.305: G#')
title('Comfortably Numb Guitar, Second Half')