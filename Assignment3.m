% AMATH 482
% Assignment 3

%% Test 1: Ideal Case
close all

load('cam1_1.mat');
[X, l1_1, means1_1] = loadMovie(vidFrames1_1);
%%    
load('cam2_1.mat');
[l2_1, means2_1] = loadMovie(vidFrames2_1);

load('cam3_1.mat');
[l3_1, means3_1] = loadMovie(vidFrames3_1);

length = min([l1_1, l2_1, l3_1]); % find the video length with the smallest number of frames
means1_1 = means1_1(1:length, :);
means2_1 = means2_1(1:length, :);
means3_1 = means3_1(1:length, :);
test1 = [means1_1'; means2_1'; means3_1'];
[m1, n1] = size(test1);
mn = mean(test1, 2);
test1 = test1-repmat(mn,1,n1);
[U1, S1, V1] = svd(test1'/sqrt(n1-1)); % perform singular value decomposition
lambda1 = diag(S1).^2; % diagonal variances
Y1 = test1' * V1; % principal components projection, plot first two

% Plot Principal Components
subplot(2,2,1)
plot(1:226, Y1(:,1), 'b', 'Linewidth', 2)
hold on
plot(1:226, Y1(:,2), 'c', 'Linewidth', 2) % plot second principal component
title('Test 1: Ideal Case', 'Fontsize', 16)
xlabel('Video Frame Iteration')
legend('First Component', 'Second Component', 'Location', 'southeast')


%% Test 2: Noisy Case

load('cam1_2.mat');
[l1_2, means1_2] = loadMovie(vidFrames1_2);

load('cam2_2.mat');
[l2_2, means2_2] = loadMovie(vidFrames2_2);

load('cam3_2.mat');
[l3_2, means3_2] = loadMovie(vidFrames3_2);

length2 = min([l1_2, l2_2, l3_2]); % find the video length with the smallest number of frames
means1_2 = means1_2(1:length2, :);
means2_2 = means2_2(1:length2, :);
means3_2 = means3_2(1:length2, :);
test2 = [means1_2'; means2_2'; means3_2'];

[m2, n2] = size(test2);
mn2 = mean(test2, 2);
test2 = test2-repmat(mn2,1,n2);
[U2, S2, V2] = svd(test2'/sqrt(n2-1)); % perform singular value decomposition
lambda2 = diag(S2).^2; % diagonal variances
Y2 = test2' * V2; % principal components projection, plot first two

% Plot Principal Components
subplot(2,2,2)
plot(1:length2, Y2(:,1), 'b', 'Linewidth', 2)
hold on
plot(1:length2, Y2(:,2), 'c', 'Linewidth', 2)
plot(1:length2, Y2(:,3), 'm', 'Linewidth', 2)
title('Test 2: Noisy Case', 'Fontsize', 16)
xlabel('Video Frame Iteration')
legend('First Component', 'Second Component', 'Third Component', 'Location', 'southeast')


%% Test 3: Horizontal Displacement

load('cam1_3.mat');
[l1_3, means1_3] = loadMovie(vidFrames1_3);

load('cam2_3.mat');
[l2_3, means2_3] = loadMovie(vidFrames2_3);

load('cam3_3.mat');
[l3_3, means3_3] = loadMovie(vidFrames3_3);

length3 = min([l1_3, l2_3, l3_3]); % find the video length with the smallest number of frames
means1_3 = means1_3(1:length3, :)';
means2_3 = means2_3(1:length3, :)';
means3_3 = means3_3(1:length3, :)';
test3 = [means1_3; means2_3; means3_3];

[m3, n3] = size(test3);
mn3 = mean(test3, 2);
test3 = test3-repmat(mn3,1,n3);
[U3, S3, V3] = svd(test3'/sqrt(n3-1)); % perform singular value decomposition
lambda3 = diag(S3).^2; % diagonal variances
Y3 = test3' * V3; % principal components projection, plot first two

% Plot Principal Components
subplot(2,2,3)
plot(1:length3, Y3(:,1), 'b', 'Linewidth', 2)
hold on
plot(1:length3, Y3(:,2), 'c', 'Linewidth', 2) % plot second principal component
plot(1:length3, Y3(:,3), 'm', 'Linewidth', 2) % third
plot(1:length3, Y3(:,4), 'g', 'Linewidth', 2)
title('Test 3: Horizontal Displacement', 'Fontsize', 16)
xlabel('Video Frame Iteration')
legend('First Component', 'Second Component', 'Third Component', 'Fourth Component', 'Location', 'southeast')


%% Test 4: Horizontal Displacement and Rotation

load('cam1_4.mat');
[l1_4, means1_4] = loadMovie(vidFrames1_4);

load('cam2_4.mat');
[l2_4, means2_4] = loadMovie(vidFrames2_4);

load('cam3_4.mat');
[l3_4, means3_4] = loadMovie(vidFrames3_4);

length4 = min([l1_4, l2_4, l3_4]); % find the video length with the smallest number of frames
means1_4 = means1_4(1:length4, :)';
means2_4 = means2_4(1:length4, :)';
means3_4 = means3_4(1:length4, :)';
test4 = [means1_4; means2_4; means3_4];

[m4, n4] = size(test4);
mn4 = mean(test4, 2);
test4 = test4-repmat(mn4,1,n4);
[U4, S4, V4] = svd(test4'/sqrt(n4-1)); % perform singular value decomposition
lambda4 = diag(S4).^2; % diagonal variances
Y4 = test4' * V4; % principal components projection, plot first two

% Plot Principal Components
subplot(2,2,4)
plot(1:length4, Y4(:,1), 'b', 'Linewidth', 2)
hold on
plot(1:length4, Y4(:,2), 'c', 'Linewidth', 2) % plot second principal component
title('Test 4: Horizontal Displacement and Rotation', 'Fontsize', 16)
xlabel('Video Frame Iteration')
legend('First Component', 'Second Component', 'Location', 'southeast')

%% Singular Value Energy Plotting
figure(2)
subplot(2,2,1)
plot(1:6, lambda1/sum(lambda1), 'r.', 'MarkerSize', 30, 'Linewidth', 2)
title('Test 1', 'Fontsize', 16)
ylabel('Energy','Fontsize', 13)
xlabel('ith Diagonal of \Sigma Matrix','Fontsize',14)
xticks([1 2 3 4 5 6])

subplot(2,2,2)
plot(1:6, lambda2/sum(lambda2), 'm.', 'MarkerSize', 30, 'Linewidth', 2)
title('Test 2', 'Fontsize', 16)
ylabel('Energy','Fontsize', 13)
xlabel('ith Diagonal of \Sigma Matrix','Fontsize',14)
xticks([1 2 3 4 5 6])

subplot(2,2,3)
plot(1:6, lambda3/sum(lambda3), 'g.', 'MarkerSize', 30, 'Linewidth', 2)
title('Test 3', 'Fontsize', 16)
ylabel('Energy','Fontsize', 13)
xlabel('ith Diagonal of \Sigma Matrix','Fontsize',14)
xticks([1 2 3 4 5 6])

subplot(2,2,4)
plot(1:6, lambda4/sum(lambda4), 'b.', 'MarkerSize', 30, 'Linewidth', 2)
title('Test 4', 'Fontsize', 16)
ylabel('Energy','Fontsize', 13)
xlabel('ith Diagonal of \Sigma Matrix','Fontsize',14)
xticks([1 2 3 4 5 6])