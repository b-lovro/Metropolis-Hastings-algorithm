function [X, acceptance_rate] = Metropolis_Hastings(num_samples, num_lags, x, sigma, burn_in)
    X = zeros(num_samples, 1); % samples of the Markov chain
    acceptance_rate = [0 0]; % vector for recording the acceptance rate

    % MH algorithm
    %%% burn-in period
    for i = 1:burn_in
        [x, a] = Metropolis_Hastings_Decision(x, sigma); 
        acceptance_rate = acceptance_rate + [a 1]; % recording the acceptance rate
    end

    %%%
    for i = 1:num_samples
        for j = 1:num_lags
            [x, a] = Metropolis_Hastings_Decision(x, sigma); 
            acceptance_rate = acceptance_rate + [a 1]; % recording the acceptance rate
        end
        X(i) = x; % save the i-th sample
    end
    
    % Plot results
    plot_results(X, acceptance_rate, num_samples);
end

function [x1, a] = Metropolis_Hastings_Decision(x0, sigma)
    xp = normrnd(x0, sigma); % generate a candidate from the normal distribution
    acceptance_probability = min(distri(xp) / distri(x0), 1); % acceptance probability
    u = rand; % random number from the interval [0,1]

    % check the acceptance criterion
    if u <= acceptance_probability 
        x1 = xp; % accept the candidate
        a = 1; % record the acceptance
    else 
        x1 = x0; % reject the candidate and keep the same point
        a = 0; % record the rejection
    end
end

function distributionX = distri(x)
    % our distribution
    distributionX = exp(-x.^2) .* (2 + sin(x * 5) + sin(x * 2));
end

function plot_results(X, acceptance_rate, num_samples)
    % Plot the samples of the Markov chain
    figure;
    subplot(2, 1, 1);
    plot(X);
    title('Samples of the Markov Chain');
    xlabel('Iteration');
    ylabel('Sample Value');
    
    % Plot the histogram of the samples and overlay the original distribution
    subplot(2, 1, 2);
    histogram(X, 30, 'Normalization', 'pdf'); % Normalize the histogram to show probability density
    hold on;
    
    % Define a range for plotting the original distribution
    x_values = linspace(min(X), max(X), 1000);
    y_values = distri(x_values);
    
    % Normalize the original distribution to match the scale of the histogram
    y_values = y_values / trapz(x_values, y_values);
    
    plot(x_values, y_values, 'r-', 'LineWidth', 2); % Plot the original distribution
    title('Histogram of Samples with Original Distribution');
    xlabel('Sample Value');
    ylabel('Density');
    legend('Samples', 'Original Distribution');
    hold off;
    
    % Calculate and display the acceptance rate
    acceptance_rate = acceptance_rate(1) / num_samples;
    fprintf('Acceptance Rate: %.2f%%\n', acceptance_rate);
end

