function rosenbrock_optimization
    % Optimization methods on the Rosenbrock function
    initial_guesses = [1.2, 1.2; -1.2, 1]; % Different starting points
    alpha0 = 1; % Initial step length
    target = [1, 1]; % Optimal point for convergence distance calculation

    for i = 1:size(initial_guesses, 1)
        initial_guess = initial_guesses(i, :);
        fprintf('Initial guess: (%.1f, %.1f)\n', initial_guess(1), initial_guess(2));

        % Run Steepest Descent
        [x_sd, alpha_sd, iter_sd] = steepest_descent(@rosenbrock_function, initial_guess, alpha0);
        fprintf('Final position for SD: (%.4f, %.4f)\n', x_sd(1), x_sd(2));

        % Run Newton's Method
        [x_newton, alpha_newton, iter_newton] = newton_method(@rosenbrock_function, initial_guess, alpha0);
        fprintf('Final position for Newton: (%.4f, %.4f)\n', x_newton(1), x_newton(2));

        % Run BFGS
        [x_bfgs, alpha_bfgs, iter_bfgs] = bfgs_method(@rosenbrock_function, initial_guess, alpha0);
        fprintf('Final position for BFGS: (%.4f, %.4f)\n', x_bfgs(1), x_bfgs(2));

        % Generate a finer grid for the Rosenbrock contour plot
[X, Y] = meshgrid(-2:0.05:2, -1:0.05:3); 
Z = 100 * (Y - X.^2).^2 + (1 - X).^2; % Rosenbrock function for contour plot

% Plot the Rosenbrock function contours
figure;
contour(X, Y, Z, 'LineColor', 'k'); 
colormap('cool');
hold on;
title('Rosenbrock Function Contours with Iterations', 'FontSize', 14);
xlabel('x', 'FontSize', 12);
ylabel('y', 'FontSize', 12);


contourf(X, Y, Z, 'LineColor', 'none'); 
alpha(0.5); 


plot(iter_sd(1:5:end,1), iter_sd(1:5:end,2), 'r-o', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'SD Trajectory'); % Steepest Descent
plot(iter_newton(1:5:end,1), iter_newton(1:5:end,2), 'g--s', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Newton Trajectory'); % Newton
plot(iter_bfgs(1:5:end,1), iter_bfgs(1:5:end,2), 'b-^', 'MarkerSize', 6, 'LineWidth', 1.5, 'DisplayName', 'BFGS Trajectory'); % BFGS

% Mark the optimum point
plot(1, 1, 'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'Optimum (1, 1)');
legend('show', 'Location', 'best');
grid on;
hold off;



        % Calculate and plot distance to optimum vs iteration in a logarithmic scale
        dist_sd = vecnorm(iter_sd - target, 2, 2);
        dist_newton = vecnorm(iter_newton - target, 2, 2);
        dist_bfgs = vecnorm(iter_bfgs - target, 2, 2);

        figure;
        semilogy(1:length(dist_sd), dist_sd, 'r-o', 'DisplayName', 'Steepest Descent');
        hold on;
        semilogy(1:length(dist_newton), dist_newton, 'g-o', 'DisplayName', 'Newton');
        semilogy(1:length(dist_bfgs), dist_bfgs, 'b-o', 'DisplayName', 'BFGS');
        xlabel('Iteration (k)');
        ylabel('Distance to Optimum (log scale)');
        title('Convergence of Methods (Log Scale)');
        legend show;
        hold off;
    end
end



function [x, alpha_k, iter] = steepest_descent(func, initial_guess, alpha0)
    x = initial_guess(:)'; % Ensure x is a 1x2 row vector
    alpha_k = zeros(1, 3000); % Preallocate for maximum iterations
    max_iter = 3000;
    tol = 1e-8; 
    iter = zeros(max_iter, 2); % Store iterations

    for k = 1:max_iter
        grad = numerical_gradient(func, x(1), x(2));
        alpha = alpha0;  % Reset step size for this iteration

        % Backtracking line search
        while func(x(1) - alpha * grad(1), x(2) - alpha * grad(2)) > func(x(1), x(2)) - 0.001 * alpha * norm(grad)^2
            alpha = alpha / 2; 
        end
        
        % Update step
        x = x - alpha * grad(:)'; % Ensure x remains a 1x2 row vector
        alpha_k(k) = alpha; % Store the step length
        iter(k, :) = x; % Store the iteration point

        % Check for convergence
        if norm(grad) < tol
            break;
        end
    end

    % Trim alpha_k and iter to actual number of iterations
    alpha_k = alpha_k(1:k);
    iter = iter(1:k, :);
end


function [x, alpha_k, iter] = newton_method(func, initial_guess, alpha0)
    x = initial_guess(:)'; % Ensure x is a 1x2 row vector
    alpha_k = zeros(1, 2000); % Preallocate for maximum iterations
    max_iter = 2000;
    tol = 1e-8;
    delta = 1e-5; % Small regularization term
    iter = zeros(max_iter, 2); % Store iterations

    for k = 1:max_iter
        grad = numerical_gradient(func, x(1), x(2));
        hess = numerical_hessian(func, x(1), x(2)) + delta * eye(2); % Regularize Hessian

        % Attempt Newton step; fallback to Steepest Descent if Hessian is not positive definite
        if any(eig(hess) <= 0)
            direction = -grad; % Steepest descent direction
        else
            direction = -hess \ grad(:); % Newton direction
        end

        % Backtracking line search
        alpha = alpha0;
        while func(x(1) + alpha * direction(1), x(2) + alpha * direction(2)) > func(x(1), x(2)) + 0.001 * alpha * (grad' * direction)
            alpha = alpha / 2;
            if alpha < 1e-14
                warning('Alpha became too small, stopping line search.');
                break;
            end
        end

        % Update step
        x = x + alpha * direction(:)';
        alpha_k(k) = alpha; % Store step length
        iter(k, :) = x; % Store iteration point

        % Check for convergence
        if norm(grad) < tol
            break;
        end
    end

    % Trim alpha_k and iter to actual number of iterations
    alpha_k = alpha_k(1:k);
    iter = iter(1:k, :);
end

function [x, alpha_k, iter] = bfgs_method(func, initial_guess, alpha0)
    x = initial_guess(:)'; % Ensure x is a 1x2 row vector
    alpha_k = zeros(1, 5000); % Increased maximum iterations
    max_iter = 5000;
    tol = 1e-8;
    B = eye(2); % Start with identity matrix as initial Hessian approximation
    iter = zeros(max_iter, 2); % Store iterations

    for k = 1:max_iter
        grad = numerical_gradient(func, x(1), x(2));
        % Compute search direction
        p = -B \ grad(:);

        % Normalize p if it is too large
        if norm(p) > 1
            p = p / norm(p);
        end

        % Adjusted backtracking line search
        alpha = alpha0 * 0.5; % Smaller initial step size
        while func(x(1) + alpha * p(1), x(2) + alpha * p(2)) > func(x(1), x(2)) + 0.0001 * alpha * (grad' * p)
            alpha = alpha * 0.9; % Smaller decrement factor
            if alpha < 1e-14 % Break if alpha becomes very small
                warning('Alpha became too small, stopping line search.');
                break;
            end
        end

        % Update x
        x_new = x + alpha * p';
        alpha_k(k) = alpha; % Store the step length
        iter(k, :) = x_new; % Store the iteration point

        % Check for convergence
        if norm(grad) < tol
            break;
        end

        % BFGS Update
        s = (x_new - x)'; % Step vector
        x = x_new; % Update x to new value
        grad_new = numerical_gradient(func, x(1), x(2));
        y = (grad_new - grad)'; % Gradient difference vector

    
        if dot(y, s) > 1e-10
            B = B + (y * y') / dot(y, s) - (B * (s * s') * B) / (s' * B * s);
        end

        % Reset B to identity every 5 iterations
        if mod(k, 5) == 0
            B = eye(2);
        end
    end

    % Trim alpha_k and iter to actual number of iterations
    alpha_k = alpha_k(1:k);
    iter = iter(1:k, :);
end


function z = rosenbrock_function(x, y)
    z = 100 * (y - x^2)^2 + (1 - x)^2;  % Rosenbrock function
end

function grad = numerical_gradient(func, x, y)
    h = 1e-6; % Small step for finite difference
    grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h);
    grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h);
    grad = [grad_x; grad_y];
end
function hess = numerical_hessian(func, x, y)
    h = 1e-6; % Small step for finite difference
    grad_x_plus_h = numerical_gradient(func, x + h, y);
    grad_x_minus_h = numerical_gradient(func, x - h, y);
    grad_y_plus_h = numerical_gradient(func, x, y + h);
    grad_y_minus_h = numerical_gradient(func, x, y - h);

    hess = zeros(2, 2);
    hess(1, 1) = (grad_x_plus_h(1) - grad_x_minus_h(1)) / (2 * h); % fxx
    hess(1, 2) = (grad_y_plus_h(1) - grad_y_minus_h(1)) / (2 * h); % fxy
    hess(2, 1) = hess(1, 2); % Symmetry
    hess(2, 2) = (grad_y_plus_h(2) - grad_y_minus_h(2)) / (2 * h); % fyy
end
