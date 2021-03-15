% AMATH 482
% Assignment 4 - Classifying Digits

% load data
[training_images, training_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

% reshape data
training = zeros(784, 60000);
for i = 1:60000
    training(:,i) = im2double(reshape(training_images(:,:,i), 784, 1));
end

% reshape data
tests = zeros(784, 10000);
for i = 1:10000
    tests(:,i) = im2double(reshape(test_images(:,:,i), 784, 1));
end

%%
% testing reshape to confirm images are the same - confirmed
for i = 1:3
subplot(2,3,i)
test = reshape(training(:,i),28,28,1);
imshow(test)
subplot(2,3,i+3)
test = training_images(:,:,i)
imshow(test)
end

%%
mn = mean(training, 2);
[m, n] = size(training);
X = training-repmat(mn, 1, n);
A = X/sqrt(n-1);

[U, S, V] = svd(A, 'econ');
lambda = diag(S).^2; % vector of singular values
subplot(1,2,1)
plot(1:784, lambda/sum(lambda)*100, 'bo'); 
title('Singular Value Energies');
xlabel('ith singular value')
ylabel('percent of total energy')

subplot(1,2,2)
singular_values = lambda/sum(lambda)*100
plot(1:154, singular_values(1:154), 'ro'); 
title('Singular Value Energies');
xlabel('ith singular value')
ylabel('percent of total energy')

%%
% Use 154 modes to represent training and test image data
projection_training = U(:, 1:154)'*X; % projection of training data
W = tests-repmat(mn, 1, 10000);
projection_test = U(:, 1:154)'*W; % projection of test data

% Finding the ideal number of singular values to capture the energy above a
% threshold
energy_threshold = 95;
total = 0;
all = sum(lambda);
diagonal = lambda/sum(lambda)*100
for j = 1:size(S)
    total = diagonal(j) + total;
    if total > energy_threshold
        break
    end
end
num_vals = j; % the number of singular values to include to capture the threshold of energy/information about image

%%
lambda = diag(S).^2;

labels = training_labels;
projection = U(:,1:154)'*X; % U is principal components, direction with most variance
scatter3(projection(1, labels == 1), projection(2, labels == 1),projection(3, labels == 1))
hold on
scatter3(projection(1, labels == 2), projection(2, labels == 2),projection(3, labels == 2))
hold on
scatter3(projection(1, labels == 3), projection(2, labels == 3),projection(3, labels == 3)) 
hold on
scatter3(projection(1, labels == 4), projection(2, labels == 4),projection(3, labels == 4))
hold on
scatter3(projection(1, labels == 5), projection(2, labels == 5),projection(3, labels == 5))
hold on
scatter3(projection(1, labels == 6), projection(2, labels == 6),projection(3, labels == 6))
hold on
scatter3(projection(1, labels == 7), projection(2, labels == 7),projection(3, labels == 7))
hold on
scatter3(projection(1, labels == 8), projection(2, labels == 8),projection(3, labels == 8),'black')
hold on
scatter3(projection(1, labels == 9), projection(2, labels == 9),projection(3, labels == 9),'red')
hold on
scatter3(projection(1, labels == 0), projection(2, labels == 0),projection(3, labels == 0),'green')
legend('1', '2', '3','4', '5','6','7','8', '9', '0', 'Fontsize', 16)
xlabel('Mode 1','Fontsize', 14)
ylabel('Mode 2','Fontsize', 14)
zlabel('Mode 3','Fontsize', 14)
%% LDA 2 Digits

feature = j; % number of features/modes to capture

im1 = training(:, training_labels == 1);
im2 = training(:, training_labels == 2);

test_1 = tests(:, test_labels == 1); % using original data - in 'tests'
test_2 = tests(:, test_labels == 2);

% redefine to cross check with training dataset

test_1 = im1;
test_2 = im2;
test_set = [test_1 test_2]; % changed to cross test on training 
hidden_labels = [zeros(1, size(test_1,2)) ones(1,size(test_2,2))];

[U,S,V,threshold,w,sortim1,sortim2] = dc_trainer(im1,im2,feature);

TestNum = size(test_set,2);
TestMat = U'*test_set; % PCA projection
pval = w'*TestMat;

ResVec = (pval>threshold);

% 0s are correct and 1s are incorrect
err = abs(ResVec - hidden_labels);
errNum = sum(err);
sucRateLDA2 = 1 - errNum/TestNum % for 1 and 2

% testing additional digits

im1 = training(:, training_labels == 4);
im2 = training(:, training_labels == 9);

test_4 = tests(:, test_labels == 4); % using original data - in 'tests'
test_9 = tests(:, test_labels == 9);
test_4 = im1;
test_9 = im2;
test_set = [test_4 test_9];
hidden_labels = [zeros(1, size(test_4,2)) ones(1,size(test_9,2))];

[U,S,V,threshold,w,sortim1,sortim2] = dc_trainer(im1,im2,feature);

TestNum = size(test_set,2);
TestMat = U'*test_set; % PCA projection
pval = w'*TestMat;

ResVec = (pval>threshold);

% 0s are correct and 1s are incorrect
err = abs(ResVec - hidden_labels);
errNum = sum(err);
sucRateLDA22 = 1 - errNum/TestNum % for 4 and 9

% testing additional digits

im1 = training(:, training_labels == 0);
im2 = training(:, training_labels == 1);

test_0 = tests(:, test_labels == 0); % using original data - in 'tests'
test_1 = tests(:, test_labels == 1);

test_0 = im1;
test_1 = im2;
test_set = [test_0 test_1];
hidden_labels = [zeros(1, size(test_0,2)) ones(1,size(test_1,2))];

[U,S,V,threshold,w,sortim1,sortim2] = dc_trainer(im1,im2,feature);

TestNum = size(test_set,2);
TestMat = U'*test_set; % PCA projection
pval = w'*TestMat;

ResVec = (pval>threshold);

% 0s are correct and 1s are incorrect
err = abs(ResVec - hidden_labels);
errNum = sum(err);
sucRateLDA23 = 1 - errNum/TestNum % for 0 and 1

%% LDA 3 digits

xtrain = projection_training(:, training_labels == 3 | training_labels == 4 | training_labels == 9);
label = training_labels(training_labels == 3 | training_labels == 4 | training_labels == 9);
label = label';
mdl = fitcdiscr(xtrain', label, 'discrimType', 'diaglinear');
test = projection_test(:, test_labels == 3 | test_labels == 4 | test_labels == 9);
label_approx = predict(mdl, test');

hidden_labels =  test_labels(test_labels == 3 | test_labels == 4 | test_labels == 9, :);
TestNum = size(hidden_labels, 1);

err = abs(label_approx - hidden_labels);
err = err > 0;
errNum = sum(err);
sucRateLDA3 = 1 - errNum/TestNum

%% LDA 3 digits method 2 - comparing error methods

xtrain = projection_training(:, training_labels == 1 | training_labels == 2 | training_labels == 3);
label = training_labels(training_labels == 1 | training_labels == 2 | training_labels == 3);
label = label';
mdl = fitcdiscr(xtrain', label, 'discrimType', 'diaglinear');
cmdl = crossval(mdl);
classErrorLDA3 = kfoldLoss(cmdl);
sucRateLDA32 = 1 - classErrorLDA3

%% SVM -  Used projection, was more accurate
xtrain = projection_training(:, training_labels == 1 | training_labels == 2);
label = training_labels(training_labels == 1 | training_labels == 2);
label = label';
Mdl = fitcsvm(xtrain',label); % svm classifier
label_approx = predict(Mdl,projection_test(:, test_labels == 1 | test_labels == 2)'); 

hidden_labels =  test_labels(test_labels == 1 | test_labels == 2, :);
TestNum = size(hidden_labels, 1);

err = abs(label_approx - hidden_labels);
err = err > 0;
errNum = sum(err);
sucRateSVM2 = 1 - errNum/TestNum

% for 4 and 9

xtrain = projection_training(:, training_labels == 4 | training_labels == 9);
label = training_labels(training_labels == 4 | training_labels == 9);
label = label';
Mdl = fitcsvm(xtrain',label); % svm classifier
label_approx = predict(Mdl,projection_test(:, test_labels == 4 | test_labels == 9)'); 

hidden_labels =  test_labels(test_labels == 4 | test_labels == 9, :);
TestNum = size(hidden_labels, 1);

err = abs(label_approx - hidden_labels);
err = err > 0;
errNum = sum(err);
sucRateSVM22 = 1 - errNum/TestNum

% for 0 and 1 - easiest 

xtrain = projection_training(:, training_labels == 0 | training_labels == 1);
label = training_labels(training_labels == 0 | training_labels == 1);
label = label';
Mdl = fitcsvm(xtrain',label); % svm classifier
label_approx = predict(Mdl,projection_test(:, test_labels == 0 | test_labels == 1)'); 

hidden_labels =  test_labels(test_labels == 0 | test_labels == 1, :);
TestNum = size(hidden_labels, 1);

err = abs(label_approx - hidden_labels);
err = err > 0;
errNum = sum(err);
sucRateSVM23 = 1 - errNum/TestNum
%% for 10 digits
 
xtrain = projection_training(:,1:10000)./max(projection_training(:,1:10000));
label = training_labels(1:10000, :);
label = label';
Mdl = fitcecoc(xtrain',label); % svm classifier
project_test = projection_test(:,1:10000)./max(projection_test(:,1:10000));
label_approx_svm = predict(Mdl,xtrain'); 

hidden_labels_svm =  test_labels(1:10000,:);
hidden_labels_svm = label'; % swapped for training comparison
TestNum = size(hidden_labels_svm, 1);

err = abs(label_approx_svm - hidden_labels_svm);
err = err > 0;
errNum = sum(err);
sucRateSVM10 = 1 - errNum/TestNum
% 
% figure()
% confusionchart(hidden_labels_svm,label_approx_svm);
% title('SVM Classifier Results for Digits 0 - 9');
%% Decision Tree - 2 digits
xtrain = projection_training(:, training_labels == 1 | training_labels == 2);
label = training_labels(training_labels == 1 | training_labels == 2);
label = label';
mdl=fitctree(xtrain',label,'MaxNumSplits', 200);
% view(mdl,'Mode','graph');
label_approx = predict(mdl, projection_test(:, test_labels == 1 | test_labels == 2)');

hidden_labels =  test_labels(test_labels == 1 | test_labels == 2, :);
TestNum = size(hidden_labels, 1);

err = abs(label_approx - hidden_labels);
err = err > 0;
errNum = sum(err);
sucRateDT2 = 1 - errNum/TestNum

% next two digit test - 4,9

xtrain = projection_training(:, training_labels == 4 | training_labels == 9);
label = training_labels(training_labels == 4 | training_labels == 9);
label = label';
mdl=fitctree(xtrain',label,'MaxNumSplits', 200);
% view(mdl,'Mode','graph');
label_approx = predict(mdl, projection_test(:, test_labels == 4 | test_labels == 9)');

hidden_labels =  test_labels(test_labels == 4 | test_labels == 9, :);
TestNum = size(hidden_labels, 1);

err = abs(label_approx - hidden_labels);
err = err > 0;
errNum = sum(err);
sucRateDT22 = 1 - errNum/TestNum

% next two digit test - 0, 1

xtrain = projection_training(:, training_labels == 0 | training_labels == 1);
label = training_labels(training_labels == 0 | training_labels == 1);
label = label';
mdl=fitctree(xtrain',label,'MaxNumSplits', 200);
% view(mdl,'Mode','graph');
label_approx = predict(mdl, projection_test(:, test_labels == 0 | test_labels == 1)');

hidden_labels =  test_labels(test_labels == 0 | test_labels == 1, :);
TestNum = size(hidden_labels, 1);

err = abs(label_approx - hidden_labels);
err = err > 0;
errNum = sum(err);
sucRateDT23 = 1 - errNum/TestNum
%% Decision Tree - 10 digits
xtrain = projection_training(:,1:10000);
label = training_labels(1:10000,:);
label = label';
mdl=fitctree(xtrain',label,'MaxNumSplits', 200)% 'CrossVal', 'on');
label_approx_dt = predict(mdl, xtrain');
hidden_labels_dt = test_labels;
hidden_labels_dt = label';

TestNum = size(hidden_labels_dt, 1);

err = abs(label_approx_dt - hidden_labels_dt);
err = err > 0;
errNum = sum(err);
sucRateDT10 = 1 - errNum/TestNum

%classErrorTree = kfoldLoss(mdl); alternative error method
%sucRateDT10 = 1- classErrorTree


figure()
confusionchart(hidden_labels_dt, label_approx_dt);
title('Decision Tree Classifier Results for Digits 0 - 9');

%% visual comparison for two digit methods
figure()
plot([1 1.5 2], [sucRateLDA2, sucRateSVM2, sucRateDT2], 'm.', 'MarkerSize', 20)
xticklabels({'LDA', 'SVM', 'DT'})
hold on
plot([1 1.5 2], [sucRateLDA22, sucRateSVM22, sucRateDT22], 'g.', 'MarkerSize', 20)
xticks([1 1.5 2])
plot([1 1.5 2], [sucRateLDA23, sucRateSVM23, sucRateDT23], 'c.', 'MarkerSize', 20)
xticklabels({'LDA', 'SVM', 'DT'})
ylim([0.9 1])
legend('1,2', '4,9', '0,1')

