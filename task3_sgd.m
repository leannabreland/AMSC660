function task3_sgd()
    close all;
    fsz = 20; 
    load('mnist.mat'); % Load the MNIST data

    % Set PCA dimensions
    nPCA = 20; 

    % Preprocess data
    [Xpca_train, label_train, ~, ~] = preprocess_data(imgs_train, labels_train, imgs_test, labels_test, nPCA);

    % Hyperparameters
    batch_sizes = [5, 16, 32, 64, 128]; 
    learning_rates = [0.001, 0.01, 0.1]; 
    max_iterations = 1000;
    tol = 1e-6; 
    dataset_size = size(Xpca_train, 1); 

    % Initialize storage for results
    results = struct();

    % Loop over batch sizes and learning rates
    for i = 1:length(batch_sizes)
        batch_size = batch_sizes(i);

        for j = 1:length(learning_rates)
            lr = learning_rates(j);

            % Fixed step size
            fprintf('\nRunning SGD with batch size %d, learning rate %.4f (fixed)\n', batch_size, lr);
            [~, loss_fixed, grad_fixed, iterations_fixed] = sgd_iterations(Xpca_train, label_train, lr, batch_size, max_iterations, tol, false, dataset_size);
            epochs_fixed = iterations_fixed / ceil(dataset_size / batch_size);
            fprintf('Fixed Step: Iterations = %d, Epochs = %.2f, Final Loss = %.4f, Final Gradient Norm = %.4e\n', ...
                    iterations_fixed, epochs_fixed, loss_fixed(end), grad_fixed(end));

            % Decreasing step size
            fprintf('Running SGD with batch size %d, learning rate %.4f (decreasing)\n', batch_size, lr);
            [~, loss_decay, grad_decay, iterations_decay] = sgd_iterations(Xpca_train, label_train, lr, batch_size, max_iterations, tol, true, dataset_size);
            epochs_decay = iterations_decay / ceil(dataset_size / batch_size);
            fprintf('Decreasing Step: Iterations = %d, Epochs = %.2f, Final Loss = %.4f, Final Gradient Norm = %.4e\n', ...
                    iterations_decay, epochs_decay, loss_decay(end), grad_decay(end));

            % Store results (replace '.' with '_')
            key = sprintf('Batch_%d_LR_%.4f', batch_size, lr);
            key = strrep(key, '.', '_'); 
            results.(key).loss_fixed = loss_fixed;
            results.(key).grad_fixed = grad_fixed;
            results.(key).iterations_fixed = iterations_fixed;
            results.(key).epochs_fixed = epochs_fixed;

            results.(key).loss_decay = loss_decay;
            results.(key).grad_decay = grad_decay;
            results.(key).iterations_decay = iterations_decay;
            results.(key).epochs_decay = epochs_decay;
        end
    end

    % Generate plots for Loss vs. Iterations
    figure;
    plot_count = 1;
    for i = 1:length(batch_sizes)
        for j = 1:length(learning_rates)
            lr = learning_rates(j);
            batch_size = batch_sizes(i);
            key = sprintf('Batch_%d_LR_%.4f', batch_size, lr);
            key = strrep(key, '.', '_'); 

            subplot(length(batch_sizes), length(learning_rates), plot_count);
            plot(1:results.(key).iterations_fixed, results.(key).loss_fixed, '-o', 'DisplayName', 'Fixed');
            hold on;
            plot(1:results.(key).iterations_decay, results.(key).loss_decay, '-x', 'DisplayName', 'Decreasing');
            title(sprintf('Batch: %d, LR: %.4f', batch_size, lr), 'FontSize', 12);
            xlabel('Iterations', 'FontSize', 10);
            ylabel('Loss', 'FontSize', 10);
            legend('Location', 'northeast', 'FontSize', 10);
            grid on;
            plot_count = plot_count + 1;
        end
    end
    sgtitle('Loss vs. Iterations for Different Configurations', 'FontSize', fsz);

    % Generate plots for Gradient Norm vs. Iterations
    figure;
    plot_count = 1;
    for i = 1:length(batch_sizes)
        for j = 1:length(learning_rates)
            lr = learning_rates(j);
            batch_size = batch_sizes(i);
            key = sprintf('Batch_%d_LR_%.4f', batch_size, lr);
            key = strrep(key, '.', '_');

            subplot(length(batch_sizes), length(learning_rates), plot_count);
            semilogy(1:results.(key).iterations_fixed, results.(key).grad_fixed, '-o', 'DisplayName', 'Fixed');
            hold on;
            semilogy(1:results.(key).iterations_decay, results.(key).grad_decay, '-x', 'DisplayName', 'Decreasing');
            title(sprintf('Batch: %d, LR: %.4f', batch_size, lr), 'FontSize', 12);
            xlabel('Iterations', 'FontSize', 10);
            ylabel('Gradient Norm', 'FontSize', 10);
            legend('Location', 'northeast', 'FontSize', 10);
            grid on;
            plot_count = plot_count + 1;
        end
    end
    sgtitle('Gradient Norm vs. Iterations for Different Configurations', 'FontSize', fsz);
end


%% Helper Function
function [w, loss, grad_norm, iterations] = sgd_iterations(X, y, lr, batch_size, max_iter, tol, decay, dataset_size)
    [n, d] = size(X);
    d2 = d^2;
    w = randn(d2 + d + 1, 1) * 0.01; 
    loss = zeros(max_iter, 1);
    grad_norm = zeros(max_iter, 1);

    for iter = 1:max_iter
        % Select a random batch
        idx = randperm(n, batch_size);
        X_batch = X(idx, :);
        y_batch = y(idx);

        % Compute residuals and gradient
        [r, J] = compute_residuals_and_jacobian(X_batch, y_batch, w);
        f = 0.5 * sum(r.^2); % Loss function
        g = J' * r;          % Gradient

        % Decay learning rate if specified
        if decay
            alpha_k = lr / (1 + iter / dataset_size);
        else
            alpha_k = lr;
        end

        % SGD update
        w = w - alpha_k * g;

        % Store loss and gradient norm
        loss(iter) = f;
        grad_norm(iter) = norm(g);

        % Check convergence
        if norm(g) < tol
            iterations = iter; 
            loss = loss(1:iter); 
            grad_norm = grad_norm(1:iter);
            return;
        end
    end
    iterations = max_iter; % If not converged within max_iter
end

function [Xpca_train, label_train, Xpca_test, label_test] = preprocess_data(imgs_train, labels_train, imgs_test, labels_test, nPCA)
    % Preprocess training and test data, apply PCA
    ind1 = find(double(labels_train) == 2);
    ind2 = find(double(labels_train) == 8);
    train1 = imgs_train(:, :, ind1);
    train2 = imgs_train(:, :, ind2);
    label_train = [ones(length(ind1), 1); -ones(length(ind2), 1)];

    % Combine training data
    X_train = [reshape(train1, [], length(ind1))'; reshape(train2, [], length(ind2))'];

    % Apply PCA
    [U, ~, ~] = svd(X_train', 'econ');
    Xpca_train = X_train * U(:, 1:nPCA);

    % Preprocess test data
    ind1_test = find(double(labels_test) == 2);
    ind2_test = find(double(labels_test) == 8);
    test1 = imgs_test(:, :, ind1_test);
    test2 = imgs_test(:, :, ind2_test);
    label_test = [ones(length(ind1_test), 1); -ones(length(ind2_test), 1)];
    X_test = [reshape(test1, [], length(ind1_test))'; reshape(test2, [], length(ind2_test))'];
    Xpca_test = X_test * U(:, 1:nPCA);
end

function [w, loss, grad_norm, iterations] = sgd(X, y, lr, batch_size, max_iter, tol)
    % Stochastic Gradient Descent with convergence check
    [n, d] = size(X);
    d2 = d^2;
    w = randn(d2 + d + 1, 1) * 0.01;
    loss = zeros(max_iter, 1);
    grad_norm = zeros(max_iter, 1);

    for iter = 1:max_iter
        % Select a random batch
        idx = randperm(n, batch_size);
        X_batch = X(idx, :);
        y_batch = y(idx);

        % Compute residuals and gradient
        [r, J] = compute_residuals_and_jacobian(X_batch, y_batch, w);
        f = 0.5 * sum(r.^2);
        g = J' * r;

        % SGD update
        w = w - lr * g;

        % Store loss and gradient norm
        loss(iter) = f;
        grad_norm(iter) = norm(g);

        % Check convergence
        if norm(g) < tol
            iterations = iter;
            loss = loss(1:iter);
            grad_norm = grad_norm(1:iter);
            return;
        end
    end
    iterations = max_iter; % If not converged within max_iter
end

function [r, J] = compute_residuals_and_jacobian(X, y, w)
    % Compute residuals and Jacobian
    [n, d] = size(X);
    d2 = d^2;
    W = reshape(w(1:d2), [d, d]);
    v = w(d2 + 1:d2 + d);
    b = w(end);

    q = diag(X * W * X') + X * v + b;
    q = y .* q;
    q = min(max(q, -50), 50);
    aux = exp(-q);
    r = log(1 + aux);

    a = -aux ./ (1 + aux);
    ya = y .* a;
    qterm = zeros(n, d2);
    for i = 1:n
        x_i = X(i, :)';
        xx = x_i * x_i';
        qterm(i, :) = xx(:)';
    end
    J = [qterm, X, ones(n, 1)];
    J = (ya * ones(1, size(J, 2))) .* J;
end
