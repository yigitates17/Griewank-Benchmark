clc;

iter_end = 200;
iter_max = 100;

min_x = -600;
max_x = 600;

syms x1 x2
gw = 1 + (1/4000)*(x1^2 + x2^2) - (cos(x1)*cos(x2/sqrt(2)));
var = [x1, x2];

grad = gradient(gw, var);
hess = hessian(gw, var);

epsilon = 10^(-4);

steepest_iter = zeros(iter_end, 1);
polak_iter = zeros(iter_end, 1);
newton_iter = zeros(iter_end, 1);
hestenes_iter = zeros(iter_end, 1);
fletcher_iter = zeros(iter_end, 1);

time_steepest = zeros(iter_end, 1);
time_fletcher = zeros(iter_end, 1);
time_hestenes = zeros(iter_end, 1);
time_polak = zeros(iter_end, 1);
time_newton = zeros(iter_end, 1);

fx_compare_steepest = zeros(iter_end, 1);
fx_compare_newton = zeros(iter_end, 1);
fx_compare_hestenes = zeros(iter_end, 1);
fx_compare_polak = zeros(iter_end, 1);
fx_compare_fletcher = zeros(iter_end, 1);

for iter = 1:iter_end

    fprintf('Iteration: %d\n', iter);

    x1_initial = (max_x - min_x) * rand() + min_x;
    x2_initial = (max_x - min_x) * rand() + min_x;
    
    x_initial = [x1_initial; x2_initial];

    if (x1_initial > max_x) || (x1_initial < min_x)
        disp("Wrong x1")
        continue
    end

    if ((x2_initial > max_x) || (x2_initial < min_x))
        disp("Wrong x2")
        continue
    end
    
    %% Steepest Descent
    fprintf('____STEEPEST DESCENT____\n');
    
    x = x_initial;
    g = double(subs(grad, var, x.'));
    
    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', 1, x(1), x(2), ...
        double(subs(gw, var, x.')), norm(g));
    
    i = 1;
    tic
    while norm(g) > epsilon

        if i > iter_max
            fprintf("Try again. Exceeded maximum number of attempts.")
            break
        end
    
        i = i + 1;
    
        % Compute gradient and Hessian at the current point
        g = double(subs(grad, var, x.'));
        H = double(subs(hess, var, x.'));
        
        % Compute the step size alpha using exact line search
        alpha = (g.' * g) / (g.' * H * g);
        
        % Update the point
        x_next = x - alpha * g;
    
        error = abs(double(subs(gw, var, x_next.')) - double(subs(gw, var, x.')));
        g_next = double(subs(grad, var, x_next.'));
    
        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, error=%f, norm=%f\n', i, x_next(1), x_next(2), ...
        double(subs(gw, var, x_next.')), error, norm(g_next));
        
        % Update x and g for the next iteration
        x = x_next;
        g = double(subs(grad, var, x.'));
        
    end
    time_steepest(iter) = toc;
    steepest_iter(iter) = i;

    fprintf("\n");

    % Display the result
    disp('Optimal point:');
    disp(x_next);

    gw_value = double(subs(gw, var, x_next.'));
    disp('Value of GW at the optimal point:');
    disp(gw_value);

    disp('Number of iterations:');
    disp(i);

    fx_compare_steepest(iter) = double(subs(gw, var, x_initial.')) - gw_value;
    
    i = 1;
    x = x_initial;

    %% Newton-Raphson

    fprintf('____NEWTON-RAPHSON____\n');

    g = double(subs(grad, var, x.'));

    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', i, x(1), x(2), ...
        double(subs(gw, var, x.')), norm(g));

    tic
    while norm(g) > epsilon

        if i > iter_max
            fprintf("Try again. Exceeded maximum number of attempts.")
            break
        end
    
        i = i + 1;

        % Compute gradient and Hessian at the current point
        g = double(subs(grad, var, x.'));
        H = double(subs(hess, var, x.'));

        % Update the point
        x_next = x - H\g;

        error = abs(double(subs(gw, var, x_next.')) - double(subs(gw, var, x.')));
        g_next = double(subs(grad, var, x_next.'));

        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, error=%f, norm=%f\n', i, x_next(1), x_next(2), ...
        double(subs(gw, var, x_next.')), error, norm(g_next));

        % Update x and g for the next iteration
        x = x_next;
        g = g_next;
        
    end
    time_newton(iter) = toc;
    newton_iter(iter) = i;

    fprintf("\n");

    % Display the result
    disp('Optimal point:');
    disp(x_next);

    gw_value = double(subs(gw, var, x_next.'));
    disp('Value of GW at the optimal point:');
    disp(gw_value);

    disp('Number of iterations:');
    disp(i);

    fx_compare_newton(iter) = double(subs(gw, var, x_initial.')) - gw_value;
    
    i = 1;
    x = x_initial;

    %% Hestenes-Stiefel

    fprintf('____HESTENES-STIEFEL____\n');

    g = double(subs(grad, var, x.'));
    d = -g;

    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', i, x(1), x(2), ...
        double(subs(gw, var, x.')), norm(g));

    tic
    while norm(g) > epsilon

        if i > iter_max
            fprintf("Try again. Exceeded maximum number of attempts.")
            break
        end
    
        i = i + 1;

        % Compute gradient and Hessian at the current point
        g = double(subs(grad, var, x.'));
        H = double(subs(hess, var, x.'));
        
        alpha = -(g.' * d)/(d.' * H * d);

        % Update the point
        x_next = x + alpha.*d;

        error = abs(double(subs(gw, var, x_next.')) - double(subs(gw, var, x.')));
        g_next = double(subs(grad, var, x_next.'));

        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, error=%f, norm=%f\n', i, x_next(1), x_next(2), ...
        double(subs(gw, var, x_next.')), error, norm(g_next));

        beta = (g_next.' * (g_next - g)) / (d.' * (g_next - g));
        d = -g_next + beta.* d;

        % Update x and g for the next iteration
        x = x_next;
        g = g_next;

    end
    time_hestenes(iter) = toc;
    hestenes_iter(iter) = i;

    fprintf("\n");

    % Display the result
    disp('Optimal point:');
    disp(x_next);

    gw_value = double(subs(gw, var, x_next.'));
    disp('Value of GW at the optimal point:');
    disp(gw_value);

    disp('Number of iterations:');
    disp(i);

    fx_compare_hestenes(iter) = double(subs(gw, var, x_initial.')) - gw_value;
    
    i = 1;
    x = x_initial;

    %% Polak-Ribiere

    fprintf('____POLAK-RIBIERE____\n');

    g = double(subs(grad, var, x.'));
    d = -g;

    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', i, x(1), x(2), ...
        double(subs(gw, var, x.')), norm(g));

    tic
    while norm(g) > epsilon

        if i > iter_max
            fprintf("Try again. Exceeded maximum number of attempts.")
            break
        end
    
        i = i + 1;

        % Compute gradient and Hessian at the current point
        g = double(subs(grad, var, x.'));
        H = double(subs(hess, var, x.'));
        
        alpha = -(g.' * d)/(d.' * H * d);

        % Update the point
        x_next = x + alpha.*d;

        error = abs(double(subs(gw, var, x_next.')) - double(subs(gw, var, x.')));
        g_next = double(subs(grad, var, x_next.'));

        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, error=%f, norm=%f\n', i, x_next(1), x_next(2), ...
        double(subs(gw, var, x_next.')), error, norm(g_next));

        beta = (g_next.' * (g_next - g)) / (g.' * g);
        d = -g_next + beta.* d;

        % Update x and g for the next iteration
        x = x_next;
        g = g_next;

    end
    time_polak(iter) = toc;
    polak_iter(iter) = i;

    fprintf("\n");

    % Display the result
    disp('Optimal point:');
    disp(x_next);

    gw_value = double(subs(gw, var, x_next.'));
    disp('Value of GW at the optimal point:');
    disp(gw_value);

    disp('Number of iterations:');
    disp(i);

    fx_compare_polak(iter) = double(subs(gw, var, x_initial.')) - gw_value;
    
    i = 1;
    x = x_initial;

    %% Fletcher-Reeves

    fprintf('____FLETCHER-REEVES____\n');

    g = double(subs(grad, var, x.'));
    d = -g;

    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', i, x(1), x(2), ...
        double(subs(gw, var, x.')), norm(g));

    tic
    while norm(g) > epsilon

        if i > iter_max
            fprintf("Try again. Exceeded maximum number of attempts.")
            break
        end
    
        i = i + 1;

        % Compute gradient and Hessian at the current point
        g = double(subs(grad, var, x.'));
        H = double(subs(hess, var, x.'));
        
        alpha = -(g.' * d)/(d.' * H * d);

        % Update the point
        x_next = x + alpha.*d;

        error = abs(double(subs(gw, var, x_next.')) - double(subs(gw, var, x.')));
        g_next = double(subs(grad, var, x_next.'));

        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, error=%f, norm=%f\n', i, x_next(1), x_next(2), ...
        double(subs(gw, var, x_next.')), error, norm(g_next));

        beta = (g_next.' * g_next) / (g.' * g);
        d = -g_next + beta.* d;

        % Update x and g for the next iteration
        x = x_next;
        g = g_next;

    end
    time_fletcher(iter) = toc;
    fletcher_iter(iter) = i;

    fprintf("\n");

    % Display the result
    disp('Optimal point:');
    disp(x_next);

    gw_value = double(subs(gw, var, x_next.'));
    disp('Value of GW at the optimal point:');
    disp(gw_value);

    disp('Number of iterations:');
    disp(i);

    fx_compare_fletcher(iter) = double(subs(gw, var, x_initial.')) - gw_value;
    
end

% Display overall results
fprintf('Steepest Descent\n');
disp(['Number of f(x_0) > f(x*): ', num2str(sum(fx_compare_steepest > 0))]);
disp(['Number of f(x_0) = f(x*): ', num2str(sum(fx_compare_steepest == 0))]);
disp(['Number of f(x_0) < f(x*): ', num2str(sum(fx_compare_steepest < 0))]);

fprintf('Newton-Raphson\n');
disp(['Number of f(x_0) > f(x*): ', num2str(sum(fx_compare_newton > 0))]);
disp(['Number of f(x_0) = f(x*): ', num2str(sum(fx_compare_newton == 0))]);
disp(['Number of f(x_0) < f(x*): ', num2str(sum(fx_compare_newton < 0))]);

fprintf('Hestenes-Stiefel\n');
disp(['Number of f(x_0) > f(x*): ', num2str(sum(fx_compare_hestenes > 0))]);
disp(['Number of f(x_0) = f(x*): ', num2str(sum(fx_compare_hestenes == 0))]);
disp(['Number of f(x_0) < f(x*): ', num2str(sum(fx_compare_hestenes < 0))]);

fprintf('Polak-Ribiere\n');
disp(['Number of f(x_0) > f(x*): ', num2str(sum(fx_compare_polak > 0))]);
disp(['Number of f(x_0) = f(x*): ', num2str(sum(fx_compare_polak == 0))]);
disp(['Number of f(x_0) < f(x*): ', num2str(sum(fx_compare_polak < 0))]);

fprintf('Fletcher-Reeves\n');
disp(['Number of f(x_0) > f(x*): ', num2str(sum(fx_compare_fletcher > 0))]);
disp(['Number of f(x_0) = f(x*): ', num2str(sum(fx_compare_fletcher == 0))]);
disp(['Number of f(x_0) < f(x*): ', num2str(sum(fx_compare_fletcher < 0))]);

fprintf('Steepest Descent Iteration Time Statistics:\n');
fprintf('Mean: %f seconds\n', mean(time_steepest));
fprintf('Minimum: %f seconds\n', min(time_steepest));
fprintf('Maximum: %f seconds\n', max(time_steepest));

fprintf('Steepest Descent Execution Time Statistics:\n');
fprintf('Mean: %f iterations\n', mean(steepest_iter));
fprintf('Minimum: %f iterations\n', min(steepest_iter));
fprintf('Maximum: %f iterations\n', max(steepest_iter));

fprintf('Newton-Raphson Iteration Time Statistics:\n');
fprintf('Mean: %f seconds\n', mean(time_newton));
fprintf('Minimum: %f seconds\n', min(time_newton));
fprintf('Maximum: %f seconds\n', max(time_newton));

fprintf('Newton-Raphson Execution Time Statistics:\n');
fprintf('Mean: %f iterations\n', mean(newton_iter));
fprintf('Minimum: %f iterations\n', min(newton_iter));
fprintf('Maximum: %f iterations\n', max(newton_iter));

fprintf('Hestenes-Stiefel Iteration Time Statistics:\n');
fprintf('Mean: %f seconds\n', mean(time_hestenes));
fprintf('Minimum: %f seconds\n', min(time_hestenes));
fprintf('Maximum: %f seconds\n', max(time_hestenes));

fprintf('Hestenes-Stiefel Execution Time Statistics:\n');
fprintf('Mean: %f iterations\n', mean(hestenes_iter));
fprintf('Minimum: %f iterations\n', min(hestenes_iter));
fprintf('Maximum: %f iterations\n', max(hestenes_iter));

fprintf('Polak-Ribiere Iteration Time Statistics:\n');
fprintf('Mean: %f seconds\n', mean(time_polak));
fprintf('Minimum: %f seconds\n', min(time_polak));
fprintf('Maximum: %f seconds\n', max(time_polak));

fprintf('Polak-Ribiere Execution Time Statistics:\n');
fprintf('Mean: %f iterations\n', mean(polak_iter));
fprintf('Minimum: %f iterations\n', min(polak_iter));
fprintf('Maximum: %f iterations\n', max(polak_iter));

fprintf('Fletcher-Reeves Iteration Time Statistics:\n');
fprintf('Mean: %f seconds\n', mean(time_fletcher));
fprintf('Minimum: %f seconds\n', min(time_fletcher));
fprintf('Maximum: %f seconds\n', max(time_fletcher));

fprintf('Fletcher-Reeves Execution Time Statistics:\n');
fprintf('Mean: %f iterations\n', mean(fletcher_iter));
fprintf('Minimum: %f iterations\n', min(fletcher_iter));
fprintf('Maximum: %f iterations\n', max(fletcher_iter));

% Boxplot for execution times
figure(1);
boxplot([time_steepest, time_newton, time_hestenes, time_polak, time_fletcher], ...
    'Labels', {'Steepest Descent', 'Newton-Raphson', 'Hestenes-Stiefel', 'Polak-Ribiere', 'Fletcher-Reeves'});
title('Execution Times for Each Algorithm');
ylabel('Time (seconds)');

% Boxplot for iteration counts
figure(2);
boxplot([steepest_iter, newton_iter, hestenes_iter, polak_iter, fletcher_iter], ...
    'Labels', {'Steepest Descent', 'Newton-Raphson', 'Hestenes-Stiefel', 'Polak-Ribiere', 'Fletcher-Reeves'});
title('Iteration Counts for Each Algorithm');
ylabel('Number of Iterations');

