function NestedDissection_anysize()
    close all;
    fsz = 10;

    %% Load the symmetric positive definite system from the maze problem
    load('maze_linear_system.mat', 'A', 'b');  
    disp('Loaded A and b from maze_linear_system.mat');

    % Pad A and b
    A = padMatrix(A);
    b = padVector(b);

    %% Compute the exact solution using MATLAB's solver
    sol = A \ b;

    %% Determine grid size from A
    n = sqrt(size(A, 1)) - 2; 
    n = round(n); 
    fprintf('Grid size inferred from A: n = %d\n', n);

    %% Nested Dissection
    level = 0;
    [L, P, A_permuted] = MyDissection(A, n + 2, n + 2, level);

    % Solve the system using the Cholesky factor and permutation
    y = L \ (P * b);         
    x_permuted = L' \ y;   
    x = P' * x_permuted;      

    % Compare the computed solution with the exact solution
    norm_diff = norm(x - sol);
    fprintf('norm(x - sol) = %e\n', norm_diff);

    % Verify that the solution satisfies A * x ≈ b
    residual_norm = norm(A * x - b);
    fprintf('Residual norm ||A*x - b|| = %e\n', residual_norm);

    % Plot the sparsity pattern of the permuted matrix
    figure;
    spy(A_permuted);
    set(gca, 'fontsize', fsz);
    grid on;
    title(sprintf('Sparsity pattern of P*A*P^T for n = %d', n + 2), 'FontSize', 20);
    axis ij;
end

function A_padded = padMatrix(A)
    % Pad A with zeros to make it 400x400
    A_padded = zeros(400, 400);
    A_padded(2:end-1, 2:end-1) = A;  % Insert A in the center
    A_padded(1, 1) = 1;               % Add 1 at the top-left corner
    A_padded(end, end) = 1;           % Add 1 at the bottom-right corner
end

function b_padded = padVector(b)
    % Pad b to match the dimensions of the new A
    b_padded = zeros(400, 1);
    b_padded(2:end-1) = b;  % Insert b in the middle
    b_padded(1) = 0;        % Add 0 at the front
    b_padded(end) = 1;      % Add 1 at the back
end

%% Recursive Nested Dissection function
function [L, P, A_permuted] = MyDissection(A, nx_grid, ny_grid, level)
    A0 = A;
    [m, ~] = size(A);

    % Base case: perform Cholesky factorization directly
    if (nx_grid <= 3) || (ny_grid <= 3)
        n_sub = size(A, 1);
        L = chol(A, 'lower');
        P = speye(n_sub);
        A_permuted = A;
        return;
    end

    N_grid = size(A, 1); % Current size of the matrix at this level
    par = mod(level, 2); % Parity to determine split direction

    % Determine split direction and compute indices
    switch par
        case 0 % Vertical split
            nx_Omega1 = floor(nx_grid / 2);
            nx_Omega3 = 1;
            nx_Omega2 = nx_grid - nx_Omega1 - nx_Omega3;
            ny_Omega1 = ny_grid;
            ny_Omega2 = ny_grid;

            % Compute indices
            [jj, ii] = meshgrid(1:nx_Omega1, 1:ny_grid);
            ind_Omega1 = sub2ind([ny_grid, nx_grid], ii(:), jj(:));
            [jj, ii] = meshgrid(nx_Omega1 + 1, 1:ny_grid);
            ind_Omega3 = sub2ind([ny_grid, nx_grid], ii(:), jj(:));
            [jj, ii] = meshgrid(nx_Omega1 + nx_Omega3 + 1:nx_grid, 1:ny_grid);
            ind_Omega2 = sub2ind([ny_grid, nx_grid], ii(:), jj(:));
        case 1 % Horizontal split
            ny_Omega1 = floor(ny_grid / 2);
            ny_Omega3 = 1; 
            ny_Omega2 = ny_grid - ny_Omega1 - ny_Omega3;
            nx_Omega1 = nx_grid;
            nx_Omega2 = nx_grid;

            % Compute indices
            [jj, ii] = meshgrid(1:nx_grid, 1:ny_Omega1);
            ind_Omega1 = sub2ind([ny_grid, nx_grid], ii(:), jj(:));
            [jj, ii] = meshgrid(1:nx_grid, ny_Omega1 + 1);
            ind_Omega3 = sub2ind([ny_grid, nx_grid], ii(:), jj(:));
            [jj, ii] = meshgrid(1:nx_grid, ny_Omega1 + ny_Omega3 + 1:ny_grid);
            ind_Omega2 = sub2ind([ny_grid, nx_grid], ii(:), jj(:));
        otherwise
            error('Invalid parity: par = %d', par);
    end

    % Adjust indices if they exceed the size of A
    ind_Omega1 = ind_Omega1(ind_Omega1 <= m);
    ind_Omega3 = ind_Omega3(ind_Omega3 <= m);
    ind_Omega2 = ind_Omega2(ind_Omega2 <= m);

    % Sizes of subdomains
    N1 = length(ind_Omega1);
    N3 = length(ind_Omega3);
    N2 = length(ind_Omega2);
    n_current = N1 + N3 + N2; % Total size at current level

    % Recursive calls
    A11 = A(ind_Omega1, ind_Omega1);
    A22 = A(ind_Omega2, ind_Omega2);

    [L11, P11, ~] = MyDissection(A11, nx_Omega1, ny_Omega1, level + 1);
    [L22, P22, ~] = MyDissection(A22, nx_Omega2, ny_Omega2, level + 1);

    % Assemble permutation matrix P1
    P1 = blkdiag(P11, speye(N3), P22);

    % Build the permutation matrix P
    P_indices = [ind_Omega1; ind_Omega3; ind_Omega2];
    P = sparse(1:n_current, P_indices, ones(n_current, 1), n_current, N_grid);
    P = P1 * P;

    % Permute A
    A = P * A0 * P';
    A_permuted = A;

    A11 = A(1:N1, 1:N1);
    A13 = A(1:N1, N1 + 1:N1 + N3);
    A31 = A(N1 + 1:N1 + N3, 1:N1);
    A22 = A(N1 + N3 + 1:end, N1 + N3 + 1:end);
    A32 = A(N1 + 1:N1 + N3, N1 + N3 + 1:end);
    A33 = A(N1 + 1:N1 + N3, N1 + 1:N1 + N3);

    L31 = L11 \ A31';
    L32 = L22 \ A32';

    % Compute Schur complement
    S33 = A33 - L31' * L31 - L32' * L32;

    % Compute Cholesky factorization of S33
    L33 = chol(S33, 'lower');

    % Assemble L
    L = blkdiag(L11, L33, L22);
    L(N1 + 1:N1 + N3, 1:N1) = L31';
    L(N1 + 1:N1 + N3, N1 + N3 + 1:end) = L32';
end
