function task2_gauss_newton()
    close all;
    fsz = 20;
    load('mnist.mat'); % Load the MNIST data

    % Set PCA dimensions
    nPCA = 20;
    [Xpca_train, label_train, Xpca_test, label_test] = preprocess_data(imgs_train, labels_train, imgs_test, labels_test, nPCA);

    % Initialize weights
    d = nPCA;
    w = ones(d^2 + d + 1, 1);

    % Gauss-Newton parameters
    max_iter = 100;
    tol = 1e-6;
    loss = zeros(max_iter, 1);
    grad_norm = zeros(max_iter, 1);

    % Iterative optimization
    for iter = 1:max_iter
        % Residuals and Jacobian
        [r, J] = compute_residuals_and_jacobian(Xpca_train, label_train, w);

        % Compute loss and gradient norm
        loss(iter) = 0.5 * sum(r.^2);
        g = J' * r;
        grad_norm(iter) = norm(g);

        % Check convergence
        if grad_norm(iter) < tol
            fprintf('Converged at iteration %d\n', iter);
            break;
        end

        % Gauss-Newton update with regularization
        H = J' * J + eye(size(J, 2)) * 1e-6;
        delta_w = -H \ g;
        w = w + delta_w;

        % Display progress
        fprintf('Iteration %d: Loss = %.6f, Gradient Norm = %.6e\n', iter, loss(iter), grad_norm(iter));
    end

    % Plot results
    figure;
    plot(1:iter, loss(1:iter), 'LineWidth', 2);
    xlabel('Iteration Number', 'FontSize', fsz);
    ylabel('Loss Function Value', 'FontSize', fsz);
    title('Loss Function vs. Iterations', 'FontSize', fsz);
    grid on;

    figure;
    semilogy(1:iter, grad_norm(1:iter), 'LineWidth', 2);
    xlabel('Iteration Number', 'FontSize', fsz);
    ylabel('Norm of Gradient', 'FontSize', fsz);
    title('Gradient Norm vs. Iterations', 'FontSize', fsz);
    grid on;
end

function [r, J] = compute_residuals_and_jacobian(X, y, w)
    % Residuals and Jacobian for the quadratic hypersurface
    [n, d] = size(X);
    d2 = d^2;
    W = reshape(w(1:d2), [d, d]);
    v = w(d2 + 1:d2 + d);
    b = w(end);

    q = diag(X * W * X') + X * v + b;
    q = y .* q; % Quadratic term with labels
    aux = exp(-q);
    r = log(1 + aux);

    % Jacobian computation
    a = -aux ./ (1 + aux);
    ya = y .* a;
    qterm = zeros(n, d2);
    for i = 1:n
        x_i = X(i, :)';
        xx = x_i * x_i'; % Outer product
        qterm(i, :) = xx(:)';
    end
    J = [qterm, X, ones(n, 1)];
    J = (ya * ones(1, size(J, 2))) .* J;
end

function [Xpca_train, label_train, Xpca_test, label_test] = preprocess_data(imgs_train, labels_train, imgs_test, labels_test, nPCA)
    % Extract and preprocess data for PCA
    ind1 = find(double(labels_train) == 2); % Class 1
    ind2 = find(double(labels_train) == 8); % Class 7
    train1 = imgs_train(:, :, ind1);
    train2 = imgs_train(:, :, ind2);
    label_train = [ones(length(ind1), 1); -ones(length(ind2), 1)];

    % Combine training data
    X_train = [reshape(train1, [], length(ind1))'; reshape(train2, [], length(ind2))'];

    % Apply PCA
    [U, ~, ~] = svd(X_train', 'econ');
    Xpca_train = X_train * U(:, 1:nPCA);

    % Preprocess test data
    ind1 = find(double(labels_test) == 2);
    ind2 = find(double(labels_test) == 8);
    test1 = imgs_test(:, :, ind1);
    test2 = imgs_test(:, :, ind2);
    label_test = [ones(length(ind1), 1); -ones(length(ind2), 1)];
    X_test = [reshape(test1, [], length(ind1))'; reshape(test2, [], length(ind2))'];
    Xpca_test = X_test * U(:, 1:nPCA);
end
