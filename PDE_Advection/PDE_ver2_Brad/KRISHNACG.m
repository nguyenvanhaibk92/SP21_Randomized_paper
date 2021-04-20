%% Regularization Parameter determination
clear all; clc;
mesh_array = [2023];
al = logspace(-6,8,30); 
% al = 2;
for n_al = 1:numel(al)
    for n = 1:numel(mesh_array)
        N = mesh_array(n);


        % --------- Initial parameters for the N mesh size -----------------------
        noise_level = 0.00001;
        load('observation.mat')
        y_obs = x';
        y_obs = y_obs + max(y_obs) * rand(size(y_obs)) * noise_level;

        load('linear_operator.mat')
        A = A; % matrix A, Operator
        SIZE_A = size(A); % size of matrix A in R:mxn
        
        load('initial_condition.mat')
        u0 = 0 *  0.25 * ones(size(A'*y_obs(:)));
        true_solution = x';

        Data_var = noise_level^2; 
        Regularization = al(n_al);

        C = 1 / Regularization * eye(SIZE_A(2));
        C_INV = Regularization * eye(SIZE_A(2));
        SIGMA = (Data_var) * eye(SIZE_A(1));
        SIGMA_INV = 1 / Data_var * eye(SIZE_A(1));
        
        
        % Solve the minimization problem using conjugate gradient method.
        % u1 = (A' * SIGMA_INV * A + C_INV) \ ((A * SIGMA_INV * y_obs(:)));
        % u1 = inv(A' * SIGMA_INV * A + C_INV) * ((A * SIGMA_INV * y_obs(:)));
        
        % Solve the minimization problem using conjugate gradient method.
        RHS = A' * (SIGMA_INV(1, 1) * y_obs(:)) + C_INV(1, 1) * u0; % Construct right hand side
        x_0 = zeros(size(RHS));
        matvecc = @(x) A' * (SIGMA_INV(1, 1) * (A * x)) + C_INV(1, 1) * x;
        max_iters = 1000; tol = 1e-8;
        u1 = CG(matvecc, RHS, x_0, max_iters, tol, true);

        error(n_al) = norm(u1 - true_solution) / norm(true_solution)*100;
    end
end
semilogx(al,error)







%% EXTRA FUNCTIONS
% ================= Function for CG =======================================
function y = LHS(x, SIGMA_INV, C_INV)
y = A' * (SIGMA_INV * (Ax_y * x)) + C_INV * x;
end

function y = LHS_RIGHT(x, SIGMA, C)
y = SIGMA * x + Ax_y(C * (A' * x));
end

function y = CG(matvec, RHS, x_0, max_iters, tol, should_print)
y = x_0;
r = RHS; % RHS - matvec(0) = RHS since matvec is linear
r_0 = r;
norm_r = norm(r_0, 2);

if tol == 0
    tol = min(0.5, norm_r) * norm(r);
else
    tol = tol * norm(r);
end

rho = r' * r;
i = 1;

while i <= max_iters && norm(r, 2) > tol
    
    if i == 1
        p = r;
    else
        beta = rho / old_rho;
        old_p = p;
        p = r + beta * p;
    end
    
    w = matvec(p);
    den = (p' * w);
    a = rho / den;
    y_old = y;
    y = y + a * p;
    r = r - a * w;
    old_rho = rho;
    rho = r' * r;
    
    %             if should_print && mod(i,5) == 0
    %                 fprintf('Iteration %4d of %4d \n',i,max_iters)
    %                 fprintf('r norm: %0.4e \n', norm(r,2))
    %             elseif mod(i,100)==0
    %                 fprintf('%4d %4d  %0.6e\n',i, max_iters, norm(r,2))
    %             end
    
    i = i + 1;
    %         [i norm(r,2) tol]
end

end
