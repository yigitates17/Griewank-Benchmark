clc

syms x1 x2
gw = 1 + (1/4000)*(x1^2 + x2^2) - (cos(x1)*cos(x2/sqrt(2)));
gw_f = @(x1_, x2_) 1 + (1/4000)*(x1_^2 + x2_^2) - (cos(x1_)*cos(x2_/sqrt(2)));
var = [x1, x2];

% Initialize x as a column vector
x_initial = [1; 1];
x = x_initial;

% Evaluate and display the initial point and its value
initial_value = double(subs(gw, var, x_initial.'));
disp('Initial point:');
disp(x_initial);
disp('Value of GW at the initial point:');
disp(initial_value);

epsilon = 10^(-4);

grad = gradient(gw, var);
hess = hessian(gw, var);

g = double(subs(grad, var, x.'));
d = -g;

i = 1;

fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, norm=%f\n', i, x(1), x(2), ...
    double(subs(gw, var, x.')), norm(g));

while norm(g) > epsilon
    
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

fprintf("\n");

% Display the result
disp('Optimal point:');
disp(x_next);

gw_value = double(subs(gw, var, x_next.'));
disp('Value of GW at the optimal point:');
disp(gw_value);

disp('Number of iterations:');
disp(i);
