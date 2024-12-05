clear;
close all;

% Define the objective function
objective = @(x, y) (x - 1).^2 + (y - 2.5).^2;

% Define the constraints
constraints = {
    @(x, y) x - 2*y + 2;          % Constraint 1: x - 2y + 2 >= 0
    @(x, y) -x - 2*y + 6;         % Constraint 2: -x - 2y + 6 >= 0
    @(x, y) -x + 2*y + 2;         % Constraint 3: -x + 2y + 2 >= 0
    @(x, y) x;                    % Constraint 4: x >= 0
    @(x, y) y                     % Constraint 5: y >= 0
};

constraint_labels = {
    'x - 2y + 2 \geq 0';
    '-x - 2y + 6 \geq 0';
    '-x + 2y + 2 \geq 0';
    'x \geq 0';
    'y \geq 0'
};

% Plotting parameters
x_vals = linspace(-1, 4, 400);
y_vals = linspace(-1, 4, 400);
[X, Y] = meshgrid(x_vals, y_vals);

figure;
hold on;
contour(X, Y, objective(X, Y), 20, 'LineWidth', 1);
title('Feasible Region and Level Sets of Objective Function');
xlabel('x');
ylabel('y');
grid on;

% Plot constraints
colors = {'r', 'g', 'b', 'c', 'm'};
for i = 1:length(constraints)
    C = constraints{i}(X, Y);
    contour(X, Y, C, [0, 0], colors{i}, 'LineWidth', 2, 'DisplayName', constraint_labels{i});
end

% Add labels to constraints
legend('show', 'Location', 'bestoutside');

% Highlight feasible region by setting constraints
feasible = ones(size(X));
for i = 1:length(constraints)
    feasible = feasible & (constraints{i}(X, Y) >= 0);
end
contourf(X, Y, feasible, [0.5, 1], 'k', 'FaceAlpha', 0.1, 'DisplayName', 'Feasible Region');

% Mark the unconstrained minimum
unconstrained_x = 1;
unconstrained_y = 2.5;
plot(unconstrained_x, unconstrained_y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Unconstrained Minimum (1, 2.5)');
text(unconstrained_x + 0.2, unconstrained_y, 'Unconstrained Minimum (1, 2.5)', 'FontSize', 10, 'Color', 'k');

hold off;

figure;
hold on;
contour(X, Y, objective(X, Y), 20, 'LineWidth', 1);
title('Feasible Region and Level Sets with Solution');
xlabel('x');
ylabel('y');
grid on;

% Plot constraints
for i = 1:length(constraints)
    C = constraints{i}(X, Y);
    contour(X, Y, C, [0, 0], colors{i}, 'LineWidth', 2, 'DisplayName', constraint_labels{i});
end

legend('show', 'Location', 'bestoutside');

% Highlight feasible region by setting constraints
contourf(X, Y, feasible, [0.5, 1], 'k', 'FaceAlpha', 0.1, 'DisplayName', 'Feasible Region');

% Add the solution point
solution_x = 1.4;
solution_y = 1.7;
plot(solution_x, solution_y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Solution (1.4, 1.7)');
text(solution_x + 0.2, solution_y, 'Solution (1.4, 1.7)', 'FontSize', 10, 'Color', 'k');
plot(unconstrained_x, unconstrained_y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Unconstrained Minimum (1, 2.5)');

hold off;
