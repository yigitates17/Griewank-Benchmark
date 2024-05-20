clc;

min_x = -600;
max_x = 600;
interval = 1;

x1 = min_x:interval:max_x;
x2 = min_x:interval:max_x;

[X1, X2] = meshgrid(x1, x2);

syms x1 x2
gw = 1 + (1/4000)*(x1^2 + x2^2) - (cos(x1)*cos(x2/sqrt(2)));
gw_f = @(x1_, x2_) 1 + (1/4000)*(x1_^2 + x2_^2) - (cos(x1_)*cos(x2_/sqrt(2)));
gw_values = arrayfun(gw_f, X1, X2);

var = [x1, x2];

grad = gradient(gw, var);
hess = hessian(gw, var);

epsilon = 10^(-4);

% Initialize x as a column vector
% x_initial = [5; 5];
x1_initial = (max_x - min_x) * rand() + min_x;
x2_initial = (max_x - min_x) * rand() + min_x; 
% x_initial = [x1_initial; x2_initial];
x_initial = [-405.3812; 353.1414];

if (x1_initial > max_x) || (x1_initial < min_x)
        disp("Wrong x1")
        return
end

if ((x2_initial > max_x) || (x2_initial < min_x))
    disp("Wrong x2")
    return
end

steepest_x = zeros(100, 2);
steepest_values = zeros(100, 1);

fletcher_x = zeros(100, 2);
fletcher_values = zeros(100, 1);

hestenes_x = zeros(100, 2);
hestenes_values = zeros(100, 1);

polak_x = zeros(100, 2);
polak_values = zeros(100, 1);

newton_x = zeros(100, 2);
newton_values = zeros(100, 1);

i = 1;
x = x_initial;

steepest_x(i, :) = x;
steepest_values(i, :) = double(subs(gw, var, x.'));

fletcher_x(i, :) = x;
fletcher_values(i, :) = double(subs(gw, var, x.'));

hestenes_x(i, :) = x;
hestenes_values(i, :) = double(subs(gw, var, x.'));

polak_x(i, :) = x;
polak_values(i, :) = double(subs(gw, var, x.'));

newton_x(i, :) = x;
newton_values(i, :) = double(subs(gw, var, x.'));

%% Steepest Descent
fprintf('____STEEPEST DESCENT____\n');

g = double(subs(grad, var, x.'));

fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', i, x(1), x(2), ...
    double(subs(gw, var, x.')), norm(g));

tic
while norm(g) > epsilon

    i = i + 1;

    % Compute gradient and Hessian at the current point
    g = double(subs(grad, var, x.'));
    H = double(subs(hess, var, x.'));
    
    alpha = (g.' * g) / (g.' * H * g);
    
    % Update the point
    x_next = x - alpha * g;

    steepest_x(i, :) = x_next;
    steepest_values(i, :) = double(subs(gw, var, x_next.'));

    error = abs(double(subs(gw, var, x_next.')) - double(subs(gw, var, x.')));
    g_next = double(subs(grad, var, x_next.'));

    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, error=%f, norm=%f\n', i, x_next(1), x_next(2), ...
    double(subs(gw, var, x_next.')), error, norm(g_next));
    
    % Update x and g for the next iteration
    x = x_next;
    g = double(subs(grad, var, x.'));
    
    
end
toc

fprintf("\n");

% Display the result
disp('Optimal point:');
disp(x_next);

gw_value = double(subs(gw, var, x_next.'));
disp('Value of GW at the optimal point:');
disp(gw_value);

disp('Number of iterations:');
disp(i);

i = 1;
x = x_initial;

%% Newton-Raphson

fprintf('____NEWTON-RAPHSON____\n');

g = double(subs(grad, var, x.'));

fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', i, x(1), x(2), ...
    double(subs(gw, var, x.')), norm(g));

tic
while norm(g) > epsilon
    
    i = i + 1;

    % Compute gradient and Hessian at the current point
    g = double(subs(grad, var, x.'));
    H = double(subs(hess, var, x.'));

    % Update the point
    x_next = x - H\g;

    newton_x(i, :) = x_next;
    newton_values(i, :) = double(subs(gw, var, x_next.'));

    error = abs(double(subs(gw, var, x_next.')) - double(subs(gw, var, x.')));
    g_next = double(subs(grad, var, x_next.'));

    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, error=%f, norm=%f\n', i, x_next(1), x_next(2), ...
    double(subs(gw, var, x_next.')), error, norm(g_next));

    % Update x and g for the next iteration
    x = x_next;
    g = g_next;
    
end
toc

fprintf("\n");

% Display the result
disp('Optimal point:');
disp(x_next);

gw_value = double(subs(gw, var, x_next.'));
disp('Value of GW at the optimal point:');
disp(gw_value);

disp('Number of iterations:');
disp(i);

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
    
    i = i + 1;

    % Compute gradient and Hessian at the current point
    g = double(subs(grad, var, x.'));
    H = double(subs(hess, var, x.'));
    
    alpha = -(g.' * d)/(d.' * H * d);

    % Update the point
    x_next = x + alpha.*d;

    hestenes_x(i, :) = x_next;
    hestenes_values(i, :) = double(subs(gw, var, x_next.'));

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
toc

fprintf("\n");

% Display the result
disp('Optimal point:');
disp(x_next);

gw_value = double(subs(gw, var, x_next.'));
disp('Value of GW at the optimal point:');
disp(gw_value);

disp('Number of iterations:');
disp(i);

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
    
    i = i + 1;

    % Compute gradient and Hessian at the current point
    g = double(subs(grad, var, x.'));
    H = double(subs(hess, var, x.'));
    
    alpha = -(g.' * d)/(d.' * H * d);

    % Update the point
    x_next = x + alpha.*d;

    polak_x(i, :) = x_next;
    polak_values(i, :) = double(subs(gw, var, x_next.'));

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
toc

fprintf("\n");

% Display the result
disp('Optimal point:');
disp(x_next);

gw_value = double(subs(gw, var, x_next.'));
disp('Value of GW at the optimal point:');
disp(gw_value);

disp('Number of iterations:');
disp(i);

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
    
    i = i + 1;

    % Compute gradient and Hessian at the current point
    g = double(subs(grad, var, x.'));
    H = double(subs(hess, var, x.'));
    
    alpha = -(g.' * d)/(d.' * H * d);

    % Update the point
    x_next = x + alpha.*d;

    fletcher_x(i, :) = x_next;
    fletcher_values(i, :) = double(subs(gw, var, x_next.'));

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
toc

fprintf("\n");

% Display the result
disp('Optimal point:');
disp(x_next);

gw_value = double(subs(gw, var, x_next.'));
disp('Value of GW at the optimal point:');
disp(gw_value);

disp('Number of iterations:');
disp(i);

% Reshape
steepest_x = reshape(nonzeros(steepest_x), [], 2);
steepest_values = nonzeros(steepest_values);

fletcher_x = reshape(nonzeros(fletcher_x), [], 2);
fletcher_values = nonzeros(fletcher_values);

hestenes_x = reshape(nonzeros(hestenes_x), [], 2);
hestenes_values = nonzeros(hestenes_values);

polak_x = reshape(nonzeros(polak_x), [], 2);
polak_values = nonzeros(polak_values);

newton_x = reshape(nonzeros(newton_x), [], 2);
newton_values = nonzeros(newton_values);

%% Contour
figure(1);
contour(X1, X2, gw_values, 50);
hold on;
xlabel('x1');
ylabel('x2');
title('Contour Plot of the Function');

plot(steepest_x(:, 1), steepest_x(:, 2), 'r-*', 'LineWidth', 1);
plot(fletcher_x(:, 1), fletcher_x(:, 2), 'b-*', 'LineWidth', 1);
plot(hestenes_x(:, 1), hestenes_x(:, 2), 'm-*', 'LineWidth', 1);
plot(polak_x(:, 1), polak_x(:, 2), 'g-*',  'LineWidth', 1);
plot(newton_x(:, 1), newton_x(:, 2), 'c-*', 'LineWidth', 1);

plot(x_initial(1), x_initial(2), '*', 'Color', 'k', 'LineWidth', 5);

% plot(steepest_x(end, 1), steepest_x(end, 2), 'r-s', 'LineWidth', 2);
% plot(fletcher_x(end, 1), fletcher_x(end, 2), 'b-s', 'LineWidth', 2);
% plot(hestenes_x(end, 1), hestenes_x(end, 2), 'm-s', 'LineWidth', 2);
% plot(polak_x(end, 1), polak_x(end, 2), 'g-s', 'LineWidth', 2);
% plot(newton_x(end, 1), newton_x(end, 2), 'c-s', 'LineWidth', 2);

% legend('Steepest Descent', 'Fletcher-Reeves', 'Hestenes-Stiefel', 'Polak-Ribiere', 'Newton-Raphson');

% Add a dummy line for the legend
h1 = plot(NaN, NaN, 'r-', 'LineWidth', 2);
h2 = plot(NaN, NaN, 'b-', 'LineWidth', 2);
h3 = plot(NaN, NaN, 'm-', 'LineWidth', 2);
h4 = plot(NaN, NaN, 'g-', 'LineWidth', 2);
h5 = plot(NaN, NaN, 'c-', 'LineWidth', 2);

% Add the legend with the custom entry
legend([h1, h2, h3, h4, h5], {'Steepest Descent', 'Fletcher-Reeves', 'Hestenes-Stiefel', 'Polak-Ribiere', 'Newton-Raphson'});

hold off;

%% Surface
figure(2);
h_surf = surf(X1, X2, gw_values, 'FaceAlpha', 0.5);
hold on;
xlabel('x1');
ylabel('x2');
zlabel('Griewank(x1, x2)');
title('Surface Plot');
plot3(steepest_x(:, 1), steepest_x(:, 2), steepest_values, 'r-*', 'LineWidth', 1);
plot3(fletcher_x(:, 1), fletcher_x(:, 2), fletcher_values, 'b-*', 'LineWidth', 1);
plot3(hestenes_x(:, 1), hestenes_x(:, 2), hestenes_values, 'm-*', 'LineWidth', 1);
plot3(polak_x(:, 1), polak_x(:, 2), polak_values, 'g-*', 'LineWidth', 1);
plot3(newton_x(:, 1), newton_x(:, 2), newton_values, 'c-*', 'LineWidth', 1);

h1 = plot(NaN, NaN, 'r-', 'LineWidth', 2);
h2 = plot(NaN, NaN, 'b-', 'LineWidth', 2);
h3 = plot(NaN, NaN, 'm-', 'LineWidth', 2);
h4 = plot(NaN, NaN, 'g-', 'LineWidth', 2);
h5 = plot(NaN, NaN, 'c-', 'LineWidth', 2);

legend([h1, h2, h3, h4, h5], {'Steepest Descent', 'Fletcher-Reeves', 'Hestenes-Stiefel', 'Polak-Ribiere', 'Newton-Raphson'});

hold off;

