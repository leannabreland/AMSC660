function monte_carlo_integration()
close all; 
clear;     
clc;   
    dims = [5, 10, 15, 20]; 
    num_samples = 1e6;      
    N_trials = 100;         
    methods = {'cube', 'ball'}; 
    
    results = struct();
    
    fprintf('Monte Carlo Integration Results:\n');
    for d = dims
        for method = methods
            [vol_mean, std_error] = monte_carlo_volume_with_error(d, num_samples, N_trials, method{1});
            fprintf('d=%d %s: Vol(B^d ∩ C^d) = %.6f ± %.6f\n', d, method{1}, vol_mean, std_error);
            results.(sprintf('d%d_%s', d, method{1})) = vol_mean;
            results.(sprintf('d%d_%s_error', d, method{1})) = std_error;
        end
    end
    
    % Volume vs. Dimension Plot
    figure;
    vol_cube = arrayfun(@(d) results.(sprintf('d%d_cube', d)), dims);
    vol_ball = arrayfun(@(d) results.(sprintf('d%d_ball', d)), dims);
    error_cube = arrayfun(@(d) results.(sprintf('d%d_cube_error', d)), dims);
    error_ball = arrayfun(@(d) results.(sprintf('d%d_ball_error', d)), dims);
    errorbar(dims, vol_cube, error_cube, '-o', 'DisplayName', 'Cube Sampling', 'LineWidth', 2);
    hold on;
    errorbar(dims, vol_ball, error_ball, '-x', 'DisplayName', 'Ball Sampling', 'LineWidth', 2);
    xlabel('Dimension d');
    ylabel('Volume of B^d ∩ C^d');
    title('Volume vs. Dimension with Standard Error');
    legend('show');
    grid on;

    % Convergence Test
    test_dimension = 10; 
    num_samples_list = logspace(3, 6, 5); 
    convergence_test(test_dimension, num_samples_list, methods, N_trials);
end

function [vol_mean, std_error] = monte_carlo_volume_with_error(d, num_samples, N_trials, method)
    volumes = zeros(N_trials, 1);
    for trial = 1:N_trials
        volumes(trial) = monte_carlo_volume(d, num_samples, method);
    end
    vol_mean = mean(volumes); 
    std_error = std(volumes) / sqrt(N_trials); 
end

function vol = monte_carlo_volume(d, num_samples, method)
    count_inside = 0; 
    if strcmp(method, 'cube')
        for i = 1:num_samples
            x = rand(d, 1) - 0.5; 
            if norm(x) <= 1
                count_inside = count_inside + 1;
            end
        end
        vol = count_inside / num_samples;
    elseif strcmp(method, 'ball')
        for i = 1:num_samples
            x = sample_from_ball(d); 
            if all(abs(x) <= 0.5)
                count_inside = count_inside + 1;
            end
        end
        vol = count_inside / num_samples * ball_volume(d);
    end
end

function x = sample_from_ball(d)
    x = randn(d, 1); 
    x = x / norm(x) * rand^(1/d); 
end

function V = ball_volume(d)
    V = pi^(d/2) / gamma(d/2 + 1);
end

function convergence_test(d, num_samples_list, methods, N_trials)
    figure;
    for method = methods
        volumes = zeros(size(num_samples_list));
        errors = zeros(size(num_samples_list));
        for i = 1:length(num_samples_list)
            num_samples = round(num_samples_list(i));
            [vol_mean, std_error] = monte_carlo_volume_with_error(d, num_samples, N_trials, method{1});
            volumes(i) = vol_mean;
            errors(i) = std_error;
        end
        errorbar(num_samples_list, volumes, errors, '-o', 'DisplayName', sprintf('%s Sampling', method{1}), 'LineWidth', 2);
        hold on;
    end
    set(gca, 'XScale', 'log'); 
    xlabel('Number of Samples');
    ylabel('Volume Estimate');
    title(sprintf('Convergence Test for d=%d with Standard Error', d));
    legend('show');
    grid on;
end
