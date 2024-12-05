% Main Script
clear;
close all;

% Load MNIST Data
load('mnist.mat');
nPCA = 20;

% Preprocess the data
[Xpca_train, label_train, Xpca_test, label_test] = preprocess_data(imgs_train, labels_train, imgs_test, labels_test, nPCA);

% Experiment Parameters
batch_sizes = [5, 16, 32, 64, 128]; 
alpha_values = [0.1, 0.01, 0.001];  % Step sizes to experiment with
epochs = 20;
lambda = 1e-4; 
tol = 1e-6; 

results = struct();

% Loop over batch sizes
for batch_size = batch_sizes
    fprintf('Running optimizers for Batch Size: %d\n', batch_size);

    % Compute iterations per epoch
    iters_per_epoch = ceil(size(Xpca_train, 1) / batch_size);

    % Loop over step sizes
    for alpha = alpha_values
        fprintf('  Step size: %.4f\n', alpha);

        % Nesterov Method
        fprintf('    Nesterov method\n');
        [loss_nesterov, grad_nesterov, misclass_nesterov] = stochastic_nesterov(Xpca_train, label_train, Xpca_test, label_test, batch_size, alpha, epochs, iters_per_epoch, lambda, tol);
        results.(sprintf('Batch_%d_Alpha_%s_Nesterov', batch_size, strrep(num2str(alpha, '%.4f'), '.', '_'))) = struct('loss', loss_nesterov, 'grad', grad_nesterov, 'misclassified', misclass_nesterov);

        % Adam Method
        fprintf('    Adam method\n');
        [loss_adam, grad_adam, misclass_adam] = stochastic_adam(Xpca_train, label_train, Xpca_test, label_test, batch_size, alpha, epochs, iters_per_epoch, lambda, tol);
        results.(sprintf('Batch_%d_Alpha_%s_Adam', batch_size, strrep(num2str(alpha, '%.4f'), '.', '_'))) = struct('loss', loss_adam, 'grad', grad_adam, 'misclassified', misclass_adam);
    end
end

% Print Convergence Results
print_results(results, batch_sizes, alpha_values);

% Visualize Results
visualize_results(results, batch_sizes, alpha_values);

%% Nesterov
function [loss, grad, misclassified] = stochastic_nesterov(X_train, y_train, X_test, y_test, batch_size, alpha, epochs, iters_per_epoch, lambda, tol)
    [n, d] = size(X_train);
    w = randn(d^2 + d + 1, 1) * 0.01; % Initialize weights
    v = zeros(size(w)); % Initialize momentum
    max_grad_norm = 10; 

    % Preallocate
    total_iters = epochs * iters_per_epoch;
    loss = zeros(total_iters, 1);
    grad = zeros(total_iters, 1);

    iter_count = 0;
    for epoch = 1:epochs
        for iter = 1:iters_per_epoch
            iter_count = iter_count + 1;

            % Select a random mini-batch
            indices = randperm(n, batch_size);
            X_batch = X_train(indices, :);
            y_batch = y_train(indices);

            % Compute gradient and loss
            [grad_k, loss_k] = compute_gradients_and_loss(w + 0.9 * v, X_batch, y_batch, lambda);

            % Check for NaNs/Infs
            if any(isnan(grad_k)) || any(isinf(grad_k))
                grad_k = zeros(size(grad_k));
            end

            % Gradient clipping
            grad_norm = norm(grad_k);
            if grad_norm > max_grad_norm
                grad_k = grad_k * (max_grad_norm / grad_norm);
            end

            % Update momentum and weights
            v = 0.9 * v - alpha * grad_k;
            w = w + v;

            % Store metrics
            loss(iter_count) = loss_k;
            grad(iter_count) = grad_norm;

            % Early stopping based on tolerance
            if grad_norm < tol
                loss = loss(1:iter_count);
                grad = grad(1:iter_count);
                misclassified = compute_misclassification(X_test, y_test, w);
                return;
            end
        end
    end

    % Compute misclassified digits
    misclassified = compute_misclassification(X_test, y_test, w);
end

%% Adam
function [loss, grad, misclassified] = stochastic_adam(X_train, y_train, X_test, y_test, batch_size, alpha, epochs, iters_per_epoch, lambda, tol)
    [n, d] = size(X_train);
    w = randn(d^2 + d + 1, 1) * 0.01; % Initialize weights
    m = zeros(size(w)); % First moment
    v = zeros(size(w)); % Second moment
    beta1 = 0.9; % Decay rate for first moment
    beta2 = 0.999; % Decay rate for second moment
    epsilon = 1e-8; % Small constant to prevent division by zero
    max_grad_norm = 10;

    % Preallocate
    total_iters = epochs * iters_per_epoch;
    loss = zeros(total_iters, 1);
    grad = zeros(total_iters, 1);

    iter_count = 0;
    for epoch = 1:epochs
        for iter = 1:iters_per_epoch
            iter_count = iter_count + 1;

            % Select a random mini-batch
            indices = randperm(n, batch_size);
            X_batch = X_train(indices, :);
            y_batch = y_train(indices);

            % Compute gradient and loss
            [grad_k, loss_k] = compute_gradients_and_loss(w, X_batch, y_batch, lambda);

            % Check for NaNs/Infs
            if any(isnan(grad_k)) || any(isinf(grad_k))
                grad_k = zeros(size(grad_k));
            end

            % Gradient clipping
            grad_norm = norm(grad_k);
            if grad_norm > max_grad_norm
                grad_k = grad_k * (max_grad_norm / grad_norm);
            end

            % Update first and second moments
            m = beta1 * m + (1 - beta1) * grad_k;
            v = beta2 * v + (1 - beta2) * (grad_k .^ 2);

            % Bias correction
            m_hat = m / (1 - beta1^iter_count);
            v_hat = v / (1 - beta2^iter_count);

            % Update weights
            w = w - alpha * m_hat ./ (sqrt(v_hat) + epsilon);

            % Store metrics
            loss(iter_count) = loss_k;
            grad(iter_count) = grad_norm;

            % Early stopping based on tolerance
            if grad_norm < tol
                loss = loss(1:iter_count);
                grad = grad(1:iter_count);
                misclassified = compute_misclassification(X_test, y_test, w);
                return;
            end
        end
    end

    % Compute misclassified digits
    misclassified = compute_misclassification(X_test, y_test, w);
end
%% Helper Functions
function misclassified = compute_misclassification(X, y, w)
    % Compute number of misclassified samples
    [n, d] = size(X);
    d2 = d^2;
    W = reshape(w(1:d2), [d, d]); % Reshape w to get W matrix
    v = w(d2 + 1:d2 + d);          % Extract v vector
    b = w(end);                    % Extract b scalar

    q = diag(X * W * X') + X * v + b;  % Quadratic function for predictions
    predictions = sign(q);  % Convert to binary predictions (-1 or 1)

    % Misclassification calculation
    misclassified = sum(predictions ~= y);  % Count the misclassified samples
end

%% Print Results
function print_results(results, batch_sizes, alpha_values)
    fprintf('\nConvergence Summary:\n');
    fprintf('%-12s %-15s %-15s %-15s %-15s %-15s\n', 'Batch Size', 'Step Size', 'Method', 'Loss', 'Misclassified', 'Grad Norm');
    for batch_size = batch_sizes
        for alpha = alpha_values
            nest_key = sprintf('Batch_%d_Alpha_%s_Nesterov', batch_size, strrep(num2str(alpha, '%.4f'), '.', '_'));
            adam_key = sprintf('Batch_%d_Alpha_%s_Adam', batch_size, strrep(num2str(alpha, '%.4f'), '.', '_'));

            % Print Nesterov Results
            if isfield(results, nest_key)
                fprintf('%-12d %-15.4f %-15s %-15.6f %-15d %-15.6f\n', batch_size, alpha, 'Nesterov', results.(nest_key).loss(end), results.(nest_key).misclassified, results.(nest_key).grad(end));
            end

            % Print Adam Results
            if isfield(results, adam_key)
                fprintf('%-12d %-15.4f %-15s %-15.6f %-15d %-15.6f\n', batch_size, alpha, 'Adam', results.(adam_key).loss(end), results.(adam_key).misclassified, results.(adam_key).grad(end));
            end
        end
    end
end
%% Visualization Function
function visualize_results(results, batch_sizes, alpha_values)
    figure;
    num_batches = length(batch_sizes);
    num_alphas = length(alpha_values);

    for i = 1:num_batches
        batch_size = batch_sizes(i);
        for j = 1:num_alphas
            alpha = alpha_values(j);
            
            nest_key = sprintf('Batch_%d_Alpha_%s_Nesterov', batch_size, strrep(num2str(alpha, '%.4f'), '.', '_'));
            adam_key = sprintf('Batch_%d_Alpha_%s_Adam', batch_size, strrep(num2str(alpha, '%.4f'), '.', '_'));

            subplot(num_batches, num_alphas, (i-1)*num_alphas + j);
            hold on;
            
            % Plot Nesterov Loss
            if isfield(results, nest_key) && isfield(results.(nest_key), 'loss')
                plot(1:length(results.(nest_key).loss), results.(nest_key).loss, '-', 'LineWidth', 2, 'DisplayName', 'Nesterov');
            end
            
            % Plot Adam Loss
            if isfield(results, adam_key) && isfield(results.(adam_key), 'loss')
                plot(1:length(results.(adam_key).loss), results.(adam_key).loss, '--', 'LineWidth', 2, 'DisplayName', 'Adam');
            end
            
            % Title and labels for the plot
            title(sprintf('Loss (Batch %d, Alpha %s)', batch_size, strrep(num2str(alpha, '%.4f'), '.', '_')));
            xlabel('Iterations');
            ylabel('Loss');
            legend('show');
            grid on;
            hold off;
        end
    end
    
    sgtitle('Comparison of Nesterov and Adam for Different Batch Sizes and Step Sizes');
end

% Preprocess Data Function
function [Xpca_train, label_train, Xpca_test, label_test] = preprocess_data(imgs_train, labels_train, imgs_test, labels_test, nPCA)
    ind1 = find(double(labels_train) == 2);  % Class 1
    ind7 = find(double(labels_train) == 8);  % Class 7

    train1 = imgs_train(:, :, ind1);
    train7 = imgs_train(:, :, ind7);
    label_train = [ones(length(ind1), 1); -ones(length(ind7), 1)];

    X_train = [reshape(train1, [], length(ind1))'; reshape(train7, [], length(ind7))'];

    [U, ~, ~] = svd(X_train', 'econ');
    Xpca_train = X_train * U(:, 1:nPCA);  % Project data onto the first nPCA components

    ind1_test = find(double(labels_test) == 2);
    ind7_test = find(double(labels_test) == 8);
    test1 = imgs_test(:, :, ind1_test);
    test7 = imgs_test(:, :, ind7_test);
    label_test = [ones(length(ind1_test), 1); -ones(length(ind7_test), 1)];
    X_test = [reshape(test1, [], length(ind1_test))'; reshape(test7, [], length(ind7_test))'];
    Xpca_test = X_test * U(:, 1:nPCA);
end

% Compute Gradients and Loss
function [grad, loss] = compute_gradients_and_loss(w, X, y, lambda)
    [n, d] = size(X);
    d2 = d^2;
    W = reshape(w(1:d2), [d, d]);
    v = w(d2 + 1:d2 + d);
    b = w(end);

    q = diag(X * W * X') + X * v + b;  % Quadratic function
    q = y .* q;  % Apply label scaling

    q = min(max(q, -50), 50);  % Limiting q(x_j; w) to range [-50, 50]

    aux = exp(-q);  % Auxiliary term for logistic function
    r = log(1 + aux);  % Residuals
    loss = mean(r) + (lambda / 2) * norm(w)^2;  % Loss with regularization

    a = -aux ./ (1 + aux);  % Gradient of residuals
    ya = y .* a;  % Apply label scaling to the gradient
    quad_terms = compute_quadratic_terms(X, ya);  % Compute quadratic terms for gradient
    linear_terms = X' * ya;  % Linear terms of the gradient
    grad = (1 / n) * [quad_terms; linear_terms; sum(ya)] + lambda * w;
end

% Compute Quadratic Terms
function quad_terms = compute_quadratic_terms(X, vec)
    [n, d] = size(X);
    quad_terms = zeros(d^2, 1);
    for i = 1:n
        xi = X(i, :)';
        xx = xi * xi';  % Outer product
        quad_terms = quad_terms + vec(i) * xx(:);
    end
end



