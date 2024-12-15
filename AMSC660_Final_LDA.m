% Load the MNIST dataset
load('mnist.mat');

imgs_test = double(imgs_test);
labels_test = double(labels_test);

[m, n, num_test] = size(imgs_test);
A = reshape(imgs_test, [m * n, num_test])'; 

% Filter for digits 3, 8, and 9
digit_indices = (labels_test == 3) | (labels_test == 8) | (labels_test == 9);
data_filtered = A(digit_indices, :);
labels_filtered = labels_test(digit_indices);

[S_w, S_b] = compute_scatter_matrices(data_filtered, labels_filtered);

L = cholesky_decomposition(S_w); 

A_sym = L \ (S_b / L'); 

% Solve the symmetric eigenvalue problem
[V_sym, D] = eig(A_sym); 

[eigenvalues, indices] = sort(diag(D), 'descend');
V_sym = V_sym(:, indices);

% Recover eigenvectors 
W = (L') \ V_sym(:, 1:2); 

data_transformed = data_filtered * W;

digit_classes = [3, 8, 9];
class_means = arrayfun(@(cls) mean(data_transformed(labels_filtered == cls, :), 1), digit_classes, 'UniformOutput', false);
class_means = cell2mat(class_means'); 

% Classify points based on nearest centroid
predicted_labels = zeros(size(labels_filtered)); 
for i = 1:length(labels_filtered)
    point = data_transformed(i, :); 
    distances = vecnorm(class_means - point, 2, 2); 
    [~, min_idx] = min(distances);
    predicted_labels(i) = digit_classes(min_idx);
end

% Count misclassified digits
num_misclassified = sum(predicted_labels ~= labels_filtered);
fprintf('Number of misclassified digits: %d\n', num_misclassified);

total_digits = length(labels_filtered);

misclassification_rate = (num_misclassified / total_digits) * 100;

fprintf('Total digits: %d\n', total_digits);
fprintf('Misclassification rate: %.2f%%\n', misclassification_rate);

% Visualization LDA
digit_classes = [3, 8, 9]; 
digit_colors = {'r', 'b', 'g'}; 

figure;
hold on;
for i = 1:length(digit_classes)
    cls = digit_classes(i);
    cls_data = data_transformed(labels_filtered == cls, :);
    scatter(cls_data(:, 1), cls_data(:, 2), 15, digit_colors{i}, 'filled'); 
end
grid on;
xlabel('LD1');
ylabel('LD2');
title('LDA Projection of MNIST Digits 3, 8, and 9');
legend(arrayfun(@(x) sprintf('Digit %d', x), digit_classes, 'UniformOutput', false)); 
hold off;

% PCA Implementation
data_centered = data_filtered - mean(data_filtered, 1);

[~, ~, V_pca] = svd(data_centered, 'econ');

% Select the top 2 right singular vectors
W_pca = V_pca(:, 1:2);

data_pca_transformed = data_centered * W_pca;

% Visualization for PCA
digit_classes = [3, 8, 9]; 
digit_colors = {'r', 'b', 'g'}; 
figure;
hold on;
for i = 1:length(digit_classes)
    cls = digit_classes(i);
    cls_data = data_pca_transformed(labels_filtered == cls, :);
    scatter(cls_data(:, 1), cls_data(:, 2), 15, digit_colors{i}, 'filled');
end
grid on;
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA Projection of MNIST Digits 3, 8, and 9');
legend(arrayfun(@(x) sprintf('Digit %d', x), digit_classes, 'UniformOutput', false)); 
hold off;

% Compare misclassification rate for PCA
class_means_pca = arrayfun(@(cls) mean(data_pca_transformed(labels_filtered == cls, :), 1), digit_classes, 'UniformOutput', false);
class_means_pca = cell2mat(class_means_pca'); 

predicted_labels_pca = zeros(size(labels_filtered)); 
for i = 1:length(labels_filtered)
    point = data_pca_transformed(i, :); 
    distances = vecnorm(class_means_pca - point, 2, 2);
    [~, min_idx] = min(distances); 
    predicted_labels_pca(i) = digit_classes(min_idx); 
end

num_misclassified_pca = sum(predicted_labels_pca ~= labels_filtered);
fprintf('Number of misclassified digits using PCA: %d\n', num_misclassified_pca);

misclassification_rate_pca = (num_misclassified_pca / total_digits) * 100;
fprintf('Misclassification rate using PCA: %.2f%%\n', misclassification_rate_pca);

function [S_w, S_b] = compute_scatter_matrices(data, labels)
    % Compute within-class (S_w) and between-class (S_b) scatter matrices
    classes = unique(labels);
    n_features = size(data, 2);
    overall_mean = mean(data, 1);

    S_w = zeros(n_features, n_features);
    S_b = zeros(n_features, n_features);

    for cls = classes'
        class_data = data(labels == cls, :);
        class_mean = mean(class_data, 1);
        n_class_samples = size(class_data, 1);

        S_w = S_w + (class_data - class_mean)' * (class_data - class_mean);

        mean_diff = (class_mean - overall_mean)';
        S_b = S_b + n_class_samples * (mean_diff * mean_diff');
    end
end

function L = cholesky_decomposition(A)
    [n, m] = size(A);
    if n ~= m
        error('Matrix must be square.');
    end
    if ~issymmetric(A)
        error('Matrix must be symmetric.');
    end

    L = zeros(n);

    for i = 1:n
        sum_diag = 0;
        for k = 1:(i-1)
            sum_diag = sum_diag + L(i, k)^2;
        end
        L(i, i) = sqrt(A(i, i) - sum_diag);

        if L(i, i) <= 0
            error('The matrix is not positive definite.');
        end

        for j = (i+1):n
            sum_offdiag = 0;
            for k = 1:(i-1)
                sum_offdiag = sum_offdiag + L(j, k) * L(i, k);
            end
            L(j, i) = (A(j, i) - sum_offdiag) / L(i, i);
        end
    end
end
