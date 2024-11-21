
close all;
clear;

% Load MNIST data
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;

% Find indices for digits 1 and 7
ind1 = find(double(labels_train) == 2);
ind2 = find(double(labels_train) == 8);
n1train = length(ind1);
n2train = length(ind2);
fprintf("There are %d '1's and %d '7's in training data\n", n1train, n2train);

itest1 = find(double(labels_test) == 2);
itest2 = find(double(labels_test) == 8);
n1test = length(itest1);
n2test = length(itest2);
fprintf("There are %d '1's and %d '7's in test data\n", n1test, n2test);

% Prepare data
train1 = imgs_train(:, :, ind1);
train2 = imgs_train(:, :, ind2);
test1 = imgs_test(:, :, itest1);
test2 = imgs_test(:, :, itest2);

% Reshape images into vectors
[d1, d2, ~] = size(train1);
X1 = reshape(train1, [], n1train)';
X2 = reshape(train2, [], n2train)';
X = [X1; X2];
Xtest1 = reshape(test1, [], n1test)';
Xtest2 = reshape(test2, [], n2test)';
Xtest = [Xtest1; Xtest2];

% Labels for training and test data
labels_train_combined = [ones(n1train, 1); -ones(n2train, 1)];
labels_test_combined = [ones(n1test, 1); -ones(n2test, 1)];

% Perform PCA
[U, Sigma, ~] = svd(X', 'econ');
esort = diag(Sigma);
figure;
plot(esort, '.', 'Markersize', 20);
xlabel('Index');
ylabel('Singular Values');
title('Singular Values of the Data');
grid;

% Define PCA dimensions to test
d_values = [5, 10, 15, 20, 25]; 
num_runs = 3; 
misclassified_counts = zeros(num_runs, length(d_values));

% Iterate over different PCA dimensions
for run = 1:num_runs
    fprintf("Run %d/%d\n", run, num_runs);
    for i = 1:length(d_values)
        nPCA = d_values(i);
        Xpca_train = X * U(:, 1:nPCA); % Project training data
        Xpca_test = Xtest * U(:, 1:nPCA); % Project test data

        % Train the quadratic model using Levenberg-Marquardt
        r_and_J = @(w) Res_and_Jac(Xpca_train, labels_train_combined, w);
        w = ones(nPCA^2 + nPCA + 1, 1); % Initialize weights
        kmax = 200; 
        tol = 1e-3; 
        [w, ~, ~] = LevenbergMarquardt(r_and_J, w, kmax, tol);

        % Test the model on the test set
        test = myquadratic(Xpca_test, labels_test_combined, w);
        hits = find(test > 0);
        misses = find(test < 0);
        misclassified_counts(run, i) = length(misses);
    end
end

% Compute average misclassifications over runs
avg_misclassified = mean(misclassified_counts);

% Plot misclassification results
figure;
plot(d_values, avg_misclassified, '-o', 'LineWidth', 2);
xlabel('Number of PCA Components');
ylabel('Average Misclassified Digits');
title('Effect of PCA Dimensions on Misclassification Rate');
grid on;

% Print results to console
disp('PCA Dimensions vs. Average Misclassified Counts:');
for i = 1:length(d_values)
    fprintf('PCA Components: %d, Avg Misclassified Digits: %.2f\n', d_values(i), avg_misclassified(i));
end

% Helper Functions from Provided Code
function f = F(r)
    f = 0.5 * r' * r;
end

function [r, J] = Res_and_Jac(X, y, w)
    aux = exp(-myquadratic(X, y, w));
    r = log(1 + aux);
    a = -aux ./ (1 + aux);
    [n, d] = size(X);
    d2 = d^2;
    ya = y .* a;
    qterm = zeros(n, d2);
    for k = 1:n
        xk = X(k, :); 
        xx = xk' * xk;
        qterm(k, :) = xx(:)';
    end
    Y = [qterm, X, ones(n, 1)];
    J = (ya * ones(1, d2 + d + 1)) .* Y;
end

function q = myquadratic(X, y, w)
    d = size(X, 2);
    d2 = d^2;
    W = reshape(w(1:d2), [d, d]);
    v = w(d2 + 1:d2 + d);
    b = w(end);
    qterm = diag(X * W * X');
    q = y .* qterm + ((y * ones(1, d)) .* X) * v + y * b;
end

function [w, fall, norg] = LevenbergMarquardt(Res_and_Jac, w, iter_max, tol)
    fsz = 16;
    [r, J] = Res_and_Jac(w);
    f = F(r);
    g = J' * r;
    nor = norm(g);
    fall = zeros(iter_max + 1, 1);
    norg = zeros(iter_max + 1, 1);
    fall(1) = f;
    norg(1) = nor;

    iter = 0;
    while nor > tol && iter < iter_max
        B = J' * J + (1e-6) * eye(length(w));
        p = -B \ g;
        w = w + p;

        [r, J] = Res_and_Jac(w);
        f = F(r);
        g = J' * r;
        nor = norm(g);

        iter = iter + 1;
        fall(iter + 1) = f;
        norg(iter + 1) = nor;
    end
    fall = fall(1:iter + 1);
    norg = norg(1:iter + 1);
end
