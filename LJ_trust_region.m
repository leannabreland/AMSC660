function LJ_trust_region()

fsz = 20; % fontsize
Na = 7; % the number of atoms
rstar = 2^(1/6); % argument of the minimum of the Lennard-Jones pair potential V(r) = r^(-12) - r^(-6)
tol = 1e-6; % stop iterations when || grad f|| < tol
iter_max = 200; % the maximal number of iterations
draw_flag = 1; % if draw_flag = 1, draw configuration at every iteration

%% parameters for trust region
Delta_max = 5; % the max trust-region radius
Delta_min = 1e-12; % the minimal trust-region radius
Delta = 1; % the initial radius
eta = 0.1; % step rejection parameter
subproblem_iter_max = 5; % the max # of iteration for quadratic subproblems
tol_sub = 1e-1; % relative tolerance for the subproblem
rho_good = 0.75;
rho_bad = 0.25;

%% Set up the initial configuration

% Define initial conditions for both predefined and random models
local_minima = {
    'Pentagonal bipyramid', 1;
    'Capped octahedron', 2;
    'Tricapped tetrahedron', 3;
    'Bicapped trigonal bipyramid', 4
};

num_random = 10;
initial_conditions = [];

% Adding predefined conditions
for i = 1:size(local_minima, 1)
    initial_conditions = [initial_conditions; local_minima(i, :)];
end

% Adding random conditions (model = 0 for random initialization)
for i = 1:num_random
    initial_conditions = [initial_conditions; {['Random initial condition #' num2str(i)], 0}];
end

%% Initialize arrays to store results
iterations_bfgs = zeros(size(initial_conditions, 1), 1);  % Pre-allocate for iteration counts
iterations_newton = zeros(size(initial_conditions, 1), 1);  % Pre-allocate for iteration counts

fvals_bfgs_all = cell(size(initial_conditions, 1), 1);  % Cell array to store function values for BFGS
fvals_newton_all = cell(size(initial_conditions, 1), 1);  % Cell array to store function values for Newton
ngvals_bfgs_all = cell(size(initial_conditions, 1), 1);  % Cell array to store gradient norms for BFGS
ngvals_newton_all = cell(size(initial_conditions, 1), 1);  % Cell array to store gradient norms for Newton

% Loop through all initial conditions
for i = 1:size(initial_conditions, 1)
    label = initial_conditions{i, 1};
    model = initial_conditions{i, 2};
    
    disp(['Running optimization for: ', label]);
    
    % Set up the initial configuration
    xyz = initial_configuration(model, Na, rstar);
    x = remove_rotations_translations(xyz);
    
    % Run BFGS optimization
    disp(['Running BFGS for ', label]);
    [iterations_bfgs(i), fvals_bfgs, ngvals_bfgs] = run_bfgs(x, Na, rstar, subproblem_iter_max, rho_bad, rho_good, Delta_max, Delta_min, eta, tol_sub, tol);
    
    % Store BFGS results
    fvals_bfgs_all{i} = fvals_bfgs;  % Store function values for BFGS
    ngvals_bfgs_all{i} = ngvals_bfgs;  % Store gradient norms for BFGS
    
    % Run Newton optimization
    disp(['Running Newton for ', label]);
    [iterations_newton(i), fvals_newton, ngvals_newton] = run_newton(x, Na, rstar, subproblem_iter_max, rho_bad, rho_good, Delta_max, Delta_min, eta, tol_sub, tol);
    
    % Store Newton results
    fvals_newton_all{i} = fvals_newton;  % Store function values for Newton
    ngvals_newton_all{i} = ngvals_newton;  % Store gradient norms for Newton
end

% Create a table for the results
result_table = table(initial_conditions(:, 1), iterations_bfgs, iterations_newton, ...
    'VariableNames', {'Initial Condition', 'BFGS Iterations', 'Newton Iterations'});

disp(result_table);

%% Plot the function values and gradient norms
% Initialize an empty cell array for the legend labels
legend_labels = {};  

% Names of the initial conditions for the legend
initial_condition_names = {
    'Pentagonal bipyramid', 
    'Capped octahedron', 
    'Tricapped tetrahedron', 
    'Bicapped trigonal bipyramid', 
    'Random initial condition #1', 
    'Random initial condition #2', 
    'Random initial condition #3', 
    'Random initial condition #4', 
    'Random initial condition #5', 
    'Random initial condition #6', 
    'Random initial condition #7', 
    'Random initial condition #8', 
    'Random initial condition #9', 
    'Random initial condition #10'
};

selected_cases = [1, 2, 3, 4, 5];

% Number of rows and columns for subplots
num_rows = length(selected_cases);
num_cols = 2; 

% Create subplots for function value and gradient norm convergence
figure;
for i = 1:length(selected_cases)
    idx = selected_cases(i);  % Get the index for the current test case
    
    fvals_bfgs_safe = max(fvals_bfgs_all{idx}, 1e-10);  
    fvals_newton_safe = max(fvals_newton_all{idx}, 1e-10);
    
    ngvals_bfgs_safe = max(ngvals_bfgs_all{idx}, 1e-10);  
    ngvals_newton_safe = max(ngvals_newton_all{idx}, 1e-10); 

    % Function value plot
    subplot(num_rows, num_cols, 2*i - 1);  % Plot on the first column of each row
    semilogy(fvals_bfgs_safe, 'r-', 'LineWidth', 2);  % BFGS function values (red), log scale
    hold on;
    semilogy(fvals_newton_safe, 'b-', 'LineWidth', 2);  % Newton function values (blue)
    xlabel('Iteration');
    ylabel('Function Value f');
    title(['Function Value Convergence - ' initial_condition_names{idx}]);
    legend({'BFGS', 'Newton'}, 'Location', 'Best');
    
    % Gradient norm plot
    subplot(num_rows, num_cols, 2*i);  % Plot on the second column of each row
    semilogy(ngvals_bfgs_safe, 'r-', 'LineWidth', 2);  % BFGS gradient norms (red), log scale
    hold on;
    semilogy(ngvals_newton_safe, 'b-', 'LineWidth', 2);  % Newton gradient norms (blue)
    xlabel('Iteration');
    ylabel('Gradient Norm ||\nabla f||');
    title(['Gradient Norm Convergence - ' initial_condition_names{idx}]);
    legend({'BFGS', 'Newton'}, 'Location', 'Best');
end

% Adjust layout
sgtitle('Convergence of BFGS and Newton for Different Initial Conditions');

end


%% BFGS and Newton Methods
function [iter_count, fvals_bfgs, ngvals_bfgs] = run_bfgs(x, Na, rstar, subproblem_iter_max, rho_bad, rho_good, Delta_max, Delta_min, eta, tol_sub, tol)
    f = LJpot(x);
    g = LJgrad(x);
    norm_g = norm(g);
    iter_max = 200;
    iter = 1;
    fvals_bfgs = zeros(iter_max, 1);
    ngvals_bfgs = zeros(iter_max, 1);
    
    fvals_bfgs(1) = f;
    ngvals_bfgs(1) = norm_g;
    
    Delta = 1;
    I = eye(length(x));  % Identity matrix
    B = I;  % Start with identity matrix for Hessian approximation
    
    while norm_g > tol && iter < iter_max
        % Compute search direction
        p = -B * g;
        p_norm = norm(p);
        
        % Check if we are on the boundary of the trust region
        if p_norm > Delta
            p = p * (Delta / p_norm);  % Scale the direction to fit the trust region
        end

        % Perform the step and update
        xnew = x + p;
        fnew = LJpot(xnew);
        gnew = LJgrad(xnew);
        
        % Compute s_k and y_k
        s_k = xnew - x;
        y_k = gnew - g;
        
        % BFGS update formula for Hessian approximation
        rho_k = 1 / (y_k' * s_k);
        B = B + rho_k * (y_k * y_k') - rho_k * (B * s_k * s_k' * B);
        
        % Regularize Hessian if it's poorly conditioned
        if rcond(B) < 1e-10
            B = B + 1e-6 * eye(size(B)); % Small diagonal regularization
        end
        
        % Store f and gradient norm
        norm_g = norm(gnew);
        
        % Update function and gradient norms for plotting
        fvals_bfgs(iter + 1) = fnew;
        ngvals_bfgs(iter + 1) = norm_g;
        
        % Update values for next iteration
        x = xnew;
        f = fnew;
        g = gnew;
        iter = iter + 1;
    end
    
    iter_count = iter;  % Return the number of iterations
end


function [iter_count, fvals_newton, ngvals_newton] = run_newton(x, Na, rstar, subproblem_iter_max, rho_bad, rho_good, Delta_max, Delta_min, eta, tol_sub, tol)
    % Initialize function value, gradient, and iteration counters
    f = LJpot(x);
    g = LJgrad(x);
    norm_g = norm(g);
    iter_max = 200;
    iter = 1;

    % Set up arrays to track function values and gradient norms
    fvals_newton = zeros(iter_max, 1);
    ngvals_newton = zeros(iter_max, 1);
    fvals_newton(1) = f;
    ngvals_newton(1) = norm_g;

    % Identity matrix for regularization
    I = eye(length(x));

    % Newton method loop
    while norm_g > tol && iter < iter_max
        % Compute Hessian and check if it is SPD (Symmetric Positive Definite)
        B = LJhess(x);
        eval_min = min(eig(B));

        % Regularize Hessian if it’s not SPD
        if eval_min <= 0
            B = B + (abs(eval_min) + 1e-6) * I;
        end

        % Solve for search direction
        p = -B \ g;

        % Perform a backtracking line search to update step length
        alpha = 1.0;
        fnew = LJpot(x + alpha * p);
        while fnew > f && alpha > 1e-10
            alpha = alpha * 0.5;
            fnew = LJpot(x + alpha * p);
        end

        % Update position, function value, gradient, and gradient norm
        x = x + alpha * p;
        f = LJpot(x);
        g = LJgrad(x);
        norm_g = norm(g);

        % Store the new function value and gradient norm
        iter = iter + 1;
        fvals_newton(iter) = f;
        ngvals_newton(iter) = norm_g;
    end

    % Return the number of iterations
    iter_count = iter;
end



%% Helper functions 
%%
function v = LJpot(xyz) % 
% xyz must be a column vector
[m, ~] = size(xyz);
Na = (m + 6)/3 ;
x_aux = [0;0;0;xyz(1);0;0;xyz(2:3);0;xyz(4:m)];
m = length(x_aux);
% restore atomic coordinates 
x = reshape(x_aux,3,Na);
r = zeros(Na);
for k = 1 : Na
    r(k,:) = sqrt(sum((x - x(:,k)*ones(1,Na)).^2,1));
end
r = r + diag(ones(Na,1));
aux = 1./r.^6;
L = (aux - 1).*aux;
L = L - diag(diag(L));
v = 2*sum(sum(L));
end

%%
function dv = LJgrad(xyz) % 
[m, ~] = size(xyz);
Na = (m + 6)/3 ;
x_aux = [0;0;0;xyz(1);0;0;xyz(2:3);0;xyz(4:m)];
m = length(x_aux);
% restore atomic coordinates
g = zeros(size(x_aux));
x = reshape(x_aux,3,Na);
r = zeros(Na);
for k = 1 : Na
    r(k,:) = sqrt(sum((x - x(:,k)*ones(1,Na)).^2,1));
end
r = r + diag(ones(Na,1));
L = -6*(2./r.^6 - 1)./r.^8;
for k = 1 : Na
    Lk = L(:,k);
    g(1 + (k-1)*3 : k*3) = (x(:,k)*ones(1,Na) - x)*Lk;
end

g = 4*g;
dv = g([4,7,8,10 : Na*3]);
end

 

% Hessian of LJ potential
function H = LJhess(x)
h = 1e-6;
n = length(x);
H = zeros(n);
e = eye(n);
for i = 1:n
    di = e(:, i) * h;
    Hei = 0.5 * (LJgrad(x + di) - LJgrad(x - di)) / h;
    for j = 1:i
        H(j, i) = e(j, :) * Hei;
        H(i, j) = H(j, i);
    end
end
H = 0.5 * (H + H');
end

% Remove rotational and translational degrees of freedom
function x = remove_rotations_translations(xyz)
[m, Na] = size(xyz);
if m ~= 3 || Na < 3
    fprintf('Error in remove_rotations_translations: [m = %d, Na = %d]\n', m, Na);
    return
end

% Shift atom 1 to the origin
xyz = xyz - xyz(:, 1) * ones(1, Na);
% Use a Householder reflection to place atom 2 on the x-axis
u = xyz(:, 2);
noru = norm(u);
ind = find(abs(u) > 1e-12, 1);
xyz = circshift(xyz, [1 - ind, 0]);
u = xyz(:, 2);
u(1) = u(1) + sign(u(1)) * noru;
noru = norm(u);
if noru > 1e-12
    House = eye(3);
    u = u / noru;
    House = House - 2 * u * u';
    xyz = House * xyz;
end
% Perform rotation around the x-axis to place atom 3 onto the xy-plane
R = eye(3);
a = xyz(:, 3);
r = sqrt(a(2)^2 + a(3)^2);
if r > 1e-12
    R(2, 2) = a(2) / r;
    R(3, 3) = R(2, 2);
    R(2, 3) = a(3) / r;
    R(3, 2) = -R(2, 3);
    xyz = R * xyz;
end

% Prepare input vector
x_aux = xyz(:); % make xyz into column vector
x = x_aux([4, 7, 8, 10:Na * 3]);
end

% Cauchy point (used for boundary cases)
function p = cauchy_point(B, g, R)
    ng = norm(g);
    ps = -g * R / ng;
    aux = g' * B * g;
    if aux <= 0
        p = ps;
    else
        a = min(ng^3 / (R * aux), 1);
        p = ps * a;
    end
end

% Initial configuration function
%% make initial configuration
function xyz = initial_configuration(model,Na,rstar)
xyz = zeros(3,Na);
switch(model)
    case 1 % Pentagonal bipyramid
        p5 = 0.4*pi;
        he = sqrt(1 - (0.5/sin(0.5*p5))^2);
        for k = 1 : 5
            xyz(1,k) = cos((k-1)*p5); 
            xyz(2,k) = sin((k-1)*p5); 
            xyz(3,k) = 0;  
        end
        xyz(3,6) = he;
        xyz(3,7) = -he;

case 2 % Capped octahedron
        r = 1/sqrt(2);
        p4 = 0.5*pi;
        pp = p4/2;
        p0 = pi*1.5 - pp;
        x0 = sin(pp);
        y0 = cos(pp);
        z0 = 0;
        for k = 1 : 4
            xyz(1,k) = x0 + r*cos(p0 + (k-1)*p4);
            xyz(2,k) = y0 + r*sin(p0 + (k-1)*p4);
            xyz(3,k) = z0;
        end
        xyz(:,5) = [x0, y0, z0 + r]';
        xyz(:,6) = [x0, y0, z0 - r]';
        xyz(:,7) = [3*x0, y0, z0 + r]';

    case 3  % Tricapped tetrahedron
        p3 = 2*pi/3;
        pp = p3/2;
        r = 1/sqrt(3);
        beta = 0.5*pi -asin(1/3) - acos(1/sqrt(3));
        r1 = cos(beta);
        p0 = 1.5*pi - pp;
        x0 = sin(pp);
        y0 = cos(pp);
        z0 = 0;
        for k = 1 : 3
            xyz(1,k) = x0 + r*cos(p0 + (k-1)*p3);
            xyz(2,k) = y0 + r*sin(p0 + (k-1)*p3);
            xyz(3,k) = z0;
            xyz(1,k + 3) = x0 + r1*cos(p0 + pp + (k-1)*p3);
            xyz(2,k + 3) = y0 + r1*sin(p0 + pp + (k-1)*p3);
            xyz(3,k + 3) = z0 + sqrt(2/3) - sin(beta);
        end
        xyz(:,7) = [x0, y0, z0 + sqrt(2/3)]';

    case 4 % Bicapped trigonal bipyramid
        p3 = 2*pi/3;
        pp = p3/2;
        r = 1/sqrt(3);
        beta = 0.5*pi -asin(1/3) - acos(1/sqrt(3));
        r1 = cos(beta);
        p0 = 1.5*pi - pp;
        x0 = sin(pp);
        y0 = cos(pp);
        z0 = 0;
        for k = 1 : 3
            xyz(1,k) = x0 + r*cos(p0 + (k-1)*p3);
            xyz(2,k) = y0 + r*sin(p0 + (k-1)*p3);
            xyz(3,k) = z0;
        end
        xyz(:,4) = [x0 + r1*cos(p0 + pp), y0 + r1*sin(p0 + pp), z0 + sqrt(2/3) - sin(beta)]';
        xyz(:,5) = [x0 + r1*cos(p0 + pp + p3), y0 + r1*sin(p0 + pp+p3), z0 - sqrt(2/3) + sin(beta)]';
        xyz(:,6) = [x0, y0, z0 - sqrt(2/3)]';
        xyz(:,7) = [x0, y0, z0 + sqrt(2/3)]';

    otherwise % random configuration
        hR = 0.01;
        xyz = zeros(3,Na);
        xyz(:,1) = [0;0;0];
        a = randn(3,Na - 1);
        rad = sqrt(a(1,:).^2 + a(2,:).^2 + a(3,:).^2);
        a = a*diag(1./rad);
        for i = 2 : Na
            clear rad
            clear x
            rad = sqrt(xyz(1,1 : i - 1).^2 + xyz(2,1 : i - 1).^2 + xyz(3,1 : i - 1).^2);
            R = max(rad) + rstar;
            xa = R*a(:,i-1);
            x = [xyz(:,1 : i - 1), xa];
            f = LJ(x(:));
            fnew = f;
            while 1
                R = R - hR;
                xa = R*a(:,i - 1);
                x = [xyz(:,1 : i - 1), xa];
                f = fnew;
                fnew = LJ(x(:));
                if fnew > f
                    break;
                end
            end
            xyz(:,i) = xa;
        end
        cmass = mean(xyz,2);
        xyz = xyz - cmass*ones(1,Na);
end
xyz = xyz*rstar;
end


%%
function v = LJ(xyz) % 
global feval
feval = feval + 1;
m = length(xyz);
Na = m/3;
x = reshape(xyz,3,Na);
r = zeros(Na);
for k = 1 : Na
    r(k,:) = sqrt(sum((x - x(:,k)*ones(1,Na)).^2,1));
end
r = r + diag(ones(Na,1));
aux = 1./r.^6;
L = (aux - 1).*aux;
L = L - diag(diag(L));
v = 2*sum(sum(L));
end