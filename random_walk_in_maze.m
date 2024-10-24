function random_walk_in_maze()
    close all
    
    %% Load maze data
    data_down = readmatrix("maze_data_down.csv");
    data_right = readmatrix("maze_data_right.csv");
    data_down(1,:) = [];
    data_down(:,1) = [];
    data_right(1,:) = [];
    data_right(:,1) = [];
    
    %% Draw the maze
    fig = 1;
    draw_maze(data_down, data_right, fig);
    
    %% Generate adjacency matrix
    [m, n] = size(data_down);
    N = m * n;
    A = make_adjacency_matrix(data_down, data_right);
    
    %% Convert adjacency matrix to stochastic matrix
    row_sums = sum(A, 2);  % Find row sums of A
    R = spdiags(1 ./ row_sums, 0, N, N);  % Diagonal matrix of inverse row sums
    P = R * A;  % Stochastic matrix for the random walk
    
    %% Setup linear system for the committor problem
    exitA = 1;
    exitB = N;
    x = zeros(N, 1);
    x(exitB) = 1;
    I = speye(N);
    L = P - I;
    b = -L * x;
    ind_exits = union(exitA, exitB);
    ind_unknown = setdiff((1:N), ind_exits);
    
    %% Make the system symmetric positive definite
    r_sqrt = sqrt(row_sums);
    D = spdiags(r_sqrt, 0, N, N);  % Matrix R^{1/2}
    Dinv = spdiags(1 ./ r_sqrt, 0, N, N);  % Matrix R^{-1/2}
    Lsymm = D * L * Dinv;
    bsymm = D * b;
    
    %% Solve the system using CG without preconditioning
    [y, ~, ~, ~, resvec] = pcg(-Lsymm(ind_unknown, ind_unknown), bsymm(ind_unknown), 1e-12, 1000);
    x(ind_unknown) = Dinv(ind_unknown, ind_unknown) * y;  % Recover the original committor
    solution_no_precond = x(ind_unknown);  % Store the solution for comparison
    
    %% Solve the system using CG with preconditioning
    ichol_fac = ichol(-Lsymm(ind_unknown, ind_unknown));
    M = ichol_fac * ichol_fac';
    
    [y_precond, ~, ~, ~, resvec_precond] = pcg(-Lsymm(ind_unknown, ind_unknown), bsymm(ind_unknown), 1e-12, 1000, M, M);
    x(ind_unknown) = Dinv(ind_unknown, ind_unknown) * y_precond;  % Recover the original committor
    solution_precond = x(ind_unknown);  % Store the solution for comparison
    
    %% Plot the residuals for CG with and without preconditioning
    figure;
    semilogy(resvec, 'b-', 'LineWidth', 2); hold on;
    semilogy(resvec_precond, 'r--', 'LineWidth', 2);
    legend('Without Preconditioning', 'With Preconditioning');
    xlabel('Iteration');
    ylabel('Residual Norm (log scale)');
    title('CG Convergence with and without Preconditioning');
    hold off;
    
    %% Visualize the committor function
    figure(3);
    committor = reshape(x, [m, n]);
    imagesc(committor);
    draw_maze(data_down, data_right, 3);
    
    %% Save the solution
    A = -Lsymm(ind_unknown, ind_unknown);
    b = -bsymm(ind_unknown);
    save("maze_linear_system.mat", "A", "b", "solution_no_precond", "solution_precond");
end

%% Function to draw the maze
function draw_maze(data_down, data_right, fig)
    [m, n] = size(data_down);
    figure(fig);
    hold on;
    line_width = 3;
    col = 'k';
    
    % Plot outer lines
    plot(0.5 + (1:n), 0.5 + zeros(1, n), 'color', col, 'Linewidth', line_width);
    plot(0.5 + (0:n-1), 0.5 + m * ones(1, n), 'color', col, 'Linewidth', line_width);
    plot(0.5 + zeros(1, m), 0.5 + (1:m), 'color', col, 'Linewidth', line_width);
    plot(0.5 + m * ones(1, n), 0.5 + (0:m-1), 'color', col, 'Linewidth', line_width);
    
    % Plot vertical lines
    for i = 1:m
        for j = 1:n-1
            if data_right(i, j) == 0
                plot(0.5 + [j, j], 0.5 + [i-1, i], 'color', col, 'Linewidth', line_width);
            end
        end
    end
    
    % Plot horizontal lines
    for j = 1:n
        for i = 1:m-1
            if data_down(i, j) == 0
                plot(0.5 + [j-1, j], 0.5 + [i, i], 'color', col, 'Linewidth', line_width);
            end
        end
    end
    
    axis ij;
    axis off;
    daspect([1, 1, 1]);
end

%% Function to make the adjacency matrix
function A = make_adjacency_matrix(data_down, data_right)
    [m, n] = size(data_down);
    mn = m * n;
    A = sparse(mn, mn);
    
    % Vertical connections (right)
    for i = 1:m
        for j = 1:n-1
            if data_right(i, j) == 1
                ind = (j-1)*m + i;
                A(ind, ind+m) = 1;
                A(ind+m, ind) = 1;
            end
        end
    end
    
    % Horizontal connections (down)
    for j = 1:n
        for i = 1:m-1
            if data_down(i, j) == 1
                ind = (j-1)*m + i;
                A(ind, ind+1) = 1;
                A(ind+1, ind) = 1;
            end
        end
    end
end
