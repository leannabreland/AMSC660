function rosenbrock_trust_region
    % Define initial guesses for optimization
    initial_guesses = [1.2, 1.2; -1.2, 1];  % Starting points
    target = [1, 1];  % Optimal point for convergence calculation

    for i = 1:size(initial_guesses, 1)
        initial_guess = initial_guesses(i, :);
        fprintf('Initial guess: (%.1f, %.1f)\n', initial_guess(1), initial_guess(2));

        % Run BFGS Trust-Region
        [x_bfgs, iter_bfgs, num_bfgs_iter] = bfgs_trust_region(@rosenbrock_function, initial_guess, 200);
        fprintf('Final position for BFGS: (%.4f, %.4f)\n', x_bfgs(1), x_bfgs(2));
        fprintf('Number of iterations for BFGS: %d\n', num_bfgs_iter);

        % Run Newton Trust-Region
        [x_newton, iter_newton, num_newton_iter] = newton_trust_region(@rosenbrock_function, initial_guess, 200);
        fprintf('Final position for Newton: (%.4f, %.4f)\n', x_newton(1), x_newton(2));
        fprintf('Number of iterations for Newton: %d\n', num_newton_iter);


        [X, Y] = meshgrid(-2:0.05:2, -1:0.05:3);
        Z = 100 * (Y - X.^2).^2 + (1 - X).^2;  % Rosenbrock function for contour

        % Plot the contours of the Rosenbrock function
        figure;
        contourf(X, Y, Z, 20, 'LineColor', 'none'); 
        colormap('cool');
        hold on;

        % Plot BFGS and Newton iteration trajectories
        plot(iter_bfgs(:,1), iter_bfgs(:,2), 'r-o', 'DisplayName', 'BFGS Trajectory');
        plot(iter_newton(:,1), iter_newton(:,2), 'b--s', 'DisplayName', 'Newton Trajectory');
        plot(1, 1, 'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'Optimum (1, 1)');
        legend show;
        title('Rosenbrock Function Contours with Iterations');
        xlabel('x');
        ylabel('y');
        hold off;

        % Calculate and plot distance to optimum
        dist_bfgs = vecnorm(iter_bfgs - target, 2, 2);
        dist_newton = vecnorm(iter_newton - target, 2, 2);

        figure;
        semilogy(1:length(dist_bfgs), dist_bfgs, 'r-o', 'DisplayName', 'BFGS');
        hold on;
        semilogy(1:length(dist_newton), dist_newton, 'b-s', 'DisplayName', 'Newton');  
        xlabel('Iteration (k)');
        ylabel('Distance to Optimum (log scale)');
        title('Convergence of Methods (Log Scale)');
        legend show;
        hold off;
    end
end

function [x, iter, num_iter] = bfgs_trust_region(func, initial_guess, max_iter)
    x = initial_guess(:)';
    tol = 1e-8;
    B = eye(2); % Start with identity matrix as initial Hessian approximation
    iter = zeros(max_iter, 2); % Store iterations
    k = 1;
    
    while k <= max_iter
        grad = numerical_gradient(func, x(1), x(2));
        s = -B \ grad(:); % Search direction
        
        % Perform a backtracking line search (or use a fixed step size for simplicity)
        alpha = 1;
        while func(x(1) + alpha * s(1), x(2) + alpha * s(2)) > func(x(1), x(2)) + 0.001 * alpha * (grad' * s)
            alpha = alpha / 2;
        end
        
        % Update x
        x_new = x + alpha * s';
        iter(k, :) = x_new;
        
        % Calculate new gradient
        grad_new = numerical_gradient(func, x_new(1), x_new(2));
        
        % Update BFGS approximation of the inverse Hessian
        y = grad_new - grad; % Gradient difference
        s = x_new - x; % Step difference
        
        % Ensure s and y are column vectors
        s = s(:); % Convert s to a column vector
        y = y(:); % Convert y to a column vector
        
        % Update B using the BFGS formula
        if (y' * s) > 1e-10 % Ensure that we have a valid update
            B = B + (y * y') / (y' * s) - (B * (s * s') * B) / (s' * B * s);
        end
        
        % Check for convergence
        if norm(grad_new) < tol
            %fprintf('Converged at iteration %d\n', k); 
            break;
        end
        
        x = x_new; 
        k = k + 1;
    end
    num_iter = k - 1;  % Record the number of iterations
end

function [x, iter, num_iter] = newton_trust_region(func, initial_guess, max_iter)
    % Trust-Region Newton with exact subspace solver
    x = initial_guess(:)';
    iter = zeros(max_iter, 2);
    tol = 1e-8;

    for k = 1:max_iter
        grad = numerical_gradient(func, x(1), x(2));  % Compute gradient
        hess = numerical_hessian(func, x(1), x(2));   % Compute Hessian

        % Newton step
        p = -hess \ grad(:);

        % Trust-region constraint: normalize step if it's too large
        delta = 1.0;
        if norm(p) > delta
            p = delta * p / norm(p);  % Apply trust-region constraint
        end

        % Update and record iteration
        x_new = x + p';
        iter(k, :) = x_new;

        % Compute new gradient at the updated step
        grad_new = numerical_gradient(func, x_new(1), x_new(2));

        x = x_new;

        % Convergence check
        if norm(grad_new) < tol
            fprintf('Converged at iteration %d\n', k);  % Show early convergence
            break;
        end
    end
    num_iter = k - 1;  % Record the number of iterations
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
