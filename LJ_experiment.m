function LJ_experiment
    % Define settings for the experiment
    directions = {'SD', 'Newton', 'BFGS_m5', 'BFGS_m20'}; % Optimization methods
    models = 1:4; % Only models 1 to 4 for known local minima
    num_tests = 10; % Number of tests for each method and model
    m_values = [5, 20]; % Reset frequencies for BFGS method
    
    % Initialize structure to store results for each method
    results = struct();
    
    for dir_idx = 1:length(directions)
        direction_name = directions{dir_idx};
        fprintf('\nStarting tests for method: %s\n', direction_name);
        
        % Initialize results field for the method
        results.(direction_name) = [];  % This works for SD and Newton

        for model = models
            num_trials = num_tests; % Always test 10 trials for known minima
            
            for test = 1:num_trials
                % Display current test information
                fprintf('Method: %s, Model: %d, Test: %d\n', direction_name, model, test);
                
                if startsWith(direction_name, 'BFGS')
                    % Extract m value from the method name
                    m = str2double(direction_name(end)); % Extract last character as number
                    % Run BFGS with specified m reset values
                    [fvals, ngvals, iter_count, final_f, final_grad_norm] = LJ_line_search(3, model, m);
                    
                    % Store results in structured format
                    result_entry = struct('fvals', fvals, 'ngvals', ngvals, 'iter_count', iter_count, ...
                                          'final_f', final_f, 'final_grad_norm', final_grad_norm, 'm', m, 'model', model);
                    results.(direction_name) = [results.(direction_name), result_entry];
                    
                    % Display final result for each test
                    fprintf('BFGS, m = %d, Model: %d, Test: %d, Iterations: %d, Final f: %f, Final ||grad f||: %f\n', ...
                            m, model, test, iter_count, final_f, final_grad_norm);
                else
                    % Run SD or Newton method
                    dir_code = find(strcmp({'SD', 'Newton'}, direction_name)); % Update this line to reference only existing methods
                    [fvals, ngvals, iter_count, final_f, final_grad_norm] = LJ_line_search(dir_code, model);
                    
                    % Store results in structured format
                    result_entry = struct('fvals', fvals, 'ngvals', ngvals, 'iter_count', iter_count, ...
                                          'final_f', final_f, 'final_grad_norm', final_grad_norm, 'model', model);
                    results.(direction_name) = [results.(direction_name), result_entry];
                    
                    % Display final result for each test
                    fprintf('Method: %s, Model: %d, Test: %d, Iterations: %d, Final f: %f, Final ||grad f||: %f\n', ...
                            direction_name, model, test, iter_count, final_f, final_grad_norm);
                end
            end
        end
    end
    
    % Plot the results for models 1 to 4
    LJ_experiment_plotting_by_model(results);
end

function LJ_experiment_plotting_by_model(results)
    % Define colors for methods
    colors = {'b', 'r', 'g', 'm', 'c', 'k'}; % Extend color options if needed
    methods = fieldnames(results); % Get the names of the methods

    % Create a figure for each model (1 to 4)
    for model = 1:4
        figure;
        sgtitle(['Comparison of Methods for Model ', num2str(model)]);

        % Subplot for Function Values
        subplot(1, 2, 1);
        hold on;
        title('Function Value');
        xlabel('Iteration');
        ylabel('f');

        % Subplot for Gradient Norm
        subplot(1, 2, 2);
        hold on;
        title('Gradient Norm');
        xlabel('Iteration');
        ylabel('||grad f||');
        set(gca, 'YScale', 'log');

        % Loop through each method
        for method_idx = 1:length(methods)
            method = methods{method_idx};
            method_results = results.(method);

            % Loop through results for the current method
            for result_idx = 1:length(method_results)
                result = method_results(result_idx);
                if result.model == model
                    % Plot function value on the first subplot
                    subplot(1, 2, 1);
                    plot(result.fvals, 'Color', colors{mod(method_idx-1, length(colors))+1}, 'DisplayName', method);
                    
                    % Plot gradient norm on the second subplot
                    subplot(1, 2, 2);
                    plot(result.ngvals, 'Color', colors{mod(method_idx-1, length(colors))+1}, 'DisplayName', method);
                end
            end
        end

        % Add legends
        subplot(1, 2, 1);
        legend('show', 'Location', 'northeast');

        subplot(1, 2, 2);
        legend('show', 'Location', 'northeast');

        hold off;
    end
end
