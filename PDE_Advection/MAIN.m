clear all; clc; close all;
rand('state', 20)
mesh_array = [2023];

for n = 1:numel(mesh_array)
    N = mesh_array(n);
    N_ITER_array = [1];% 10 100]; % number of iterations 1/10/100
    r_array = ceil([N/10]);% N/5 N/2]); % 10 / 20 / 50 percent of mesh size
    
    % --------- Initial parameters for the N mesh size -----------------------
    noise_level = 0.01;
    load('OBSERVATION_ORIGINAL.mat')
    y_obs = x';
    y_obs = y_obs + max(y_obs) * rand(size(y_obs)) * noise_level;
    
    load('LINEAR_OPERATOR.mat')
    A = AA; % matrix A, Operator
    SIZE_A = size(A); % size of matrix A in R:mxn
    
    load('INITIAL_CONDITION_ORIGINAL.mat')
    true_solution = initial_condition;
    
    Data_var = noise_level^2; Regularization = 0.1;
    
    C = 1 / Regularization * eye(SIZE_A(2));
    C_INV = Regularization * eye(SIZE_A(2));
    SIGMA = (Data_var) * eye(SIZE_A(1));
    SIGMA_INV = 1 / Data_var * eye(SIZE_A(1));
    
    max_iters = 500; tol = 1e-8;
    
    % u1 = inv(A' * SIGMA_INV * A + C_INV) * ((A * SIGMA_INV * y_obs(:)));
    % Solve the minimization problem using conjugate gradient method.
    RHS = A' * (SIGMA_INV(1, 1) * y_obs(:)); % Construct right hand side
    x_0 = zeros(size(RHS));
    matvecc = @(x) A' * (SIGMA_INV(1, 1) * (A * x)) + C_INV(1, 1) * x;
    u1 = CG(matvecc, RHS, x_0, max_iters, tol, false);
    
    % u2 = (C*A')*(inv(SIGMA + A*C*A') * y_obs);
    RHS = y_obs(:); x_0 = zeros(size(RHS));
    matvecc = @(x) SIGMA(1, 1) * x + A * (C(1, 1) * (A' * x));
    Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
    u2 = C(1, 1) * (A' * (Y));
    
    % -------------------------------------------------------------------------
    
    result_RIGHT = zeros(N, numel(r_array), numel(N_ITER_array));
    result_LEFT = zeros(N, numel(r_array), numel(N_ITER_array));
    result_RAN_MAP = zeros(N, numel(r_array), numel(N_ITER_array));
    
    for iter = 1:length(N_ITER_array)
        N_ITER = N_ITER_array(iter);
        
        for i = 1:length(r_array)
            r = r_array(i);
            [N_ITER r]
            
            
            result_RIGHT_in_ITER = zeros(N, 1);
            result_LEFT_in_ITER = zeros(N, 1);
            result_RAN_MAP_in_ITER = zeros(N, 1);
            
            for j = 1:N_ITER
                %% Using rank one product
                EPSILON = normrnd(0, sqrt(1 / Regularization), [SIZE_A(2), r]);
                LAMBDA = normrnd(0, sqrt(1 / Data_var), [size(y_obs(:), 1), r]);
                sig_rand = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), r]);
                
                %% =================== LEFT SKETCHING =====================
                SIGMA_INV_rand = 1 / r * LAMBDA * LAMBDA';
                % Solve the minimization problem using conjugate gradient method.
                RHS = A' * (SIGMA_INV_rand * y_obs(:)); % Construct right hand side
                x_0 = zeros(size(RHS));
                matvecc = @(x) A' * (SIGMA_INV_rand * (A * x)) + C_INV * x;
                result_LEFT_in_ITER = result_LEFT_in_ITER + 1 / N_ITER * ...
                    CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                %% ============= RANDOMIZED MAP solution ==================
                % Solve the minimization problem using conjugate gradient method.
                RHS = A' * (SIGMA_INV(1, 1) * (y_obs(:) + mean(sig_rand, 2))) + C_INV(1, 1) * mean(EPSILON, 2);
                x_0 = zeros(size(RHS));
                matvecc = @(x) A' * (SIGMA_INV(1, 1) * (A * x)) + C_INV(1, 1) * x;
                result_RAN_MAP_in_ITER = result_RAN_MAP_in_ITER + 1 / N_ITER * ...
                    CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                %% ============ RIGHT SKETCHING ===========================
                C_RAND = 1 / r * EPSILON * EPSILON';
                % (1) using CG for solvong Y = (SIGMA + A C A')^{-1} d
                % => (2) Then u_RS  = C A' Y
                % (1) -----------------------------------------------------
                RHS = y_obs(:);
                x_0 = zeros(size(RHS));
                matvecc = @(x) SIGMA(1, 1) * x + A * (C_RAND * (A' * x));
                Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
                % (2) -----------------------------------------------------
                u_RS = C_RAND * (A' * (Y));
                result_RIGHT_in_ITER = result_RIGHT_in_ITER + 1 / N_ITER * u_RS;
                
            end
            Error_Right(i, iter) = norm(result_RIGHT_in_ITER - u2) / norm(u2);
            Error_LEFT(i, iter) = norm(result_LEFT_in_ITER - u1) / norm(u1);
            Error_RAN_MAP(i, iter) = norm(result_RAN_MAP_in_ITER - u1) / norm(u1);
            
            result_RIGHT(:, i, iter) = result_RIGHT_in_ITER;
            result_LEFT(:, i, iter) = result_LEFT_in_ITER;
            result_RAN_MAP(:, i, iter) = result_RAN_MAP_in_ITER;
            
        end
    end

end

save ('Result', 'result_LEFT','result_RIGHT', 'result_RAN_MAP', 'Error_LEFT', 'Error_Right', 'Error_RAN_MAP');







% %% Regularization Parameter determination
% clear all; clc;
% al = logspace(-3,1,20); mesh_array = [2023];
% for n_al = 1:numel(al)
%     for n = 1:numel(mesh_array)
%         N = mesh_array(n);
%
%
%         % --------- Initial parameters for the N mesh size -----------------------
%         noise_level = 0.01;
%         load('OBSERVATION_ORIGINAL.mat')
%         y_obs = x';
%         y_obs = y_obs + max(y_obs) * rand(size(y_obs)) * noise_level;
%
%         load('LINEAR_OPERATOR.mat')
%         A = AA; % matrix A, Operator
%         SIZE_A = size(A); % size of matrix A in R:mxn
%
%         load('INITIAL_CONDITION_ORIGINAL.mat')
%         true_solution = initial_condition;
%
%         Data_var = noise_level^2; Regularization = al(n_al);
%
%         C = 1 / Regularization * eye(SIZE_A(2));
%         C_INV = Regularization * eye(SIZE_A(2));
%         SIGMA = (Data_var) * eye(SIZE_A(1));
%         SIGMA_INV = 1 / Data_var * eye(SIZE_A(1));
%
% %         % Solve the minimization problem using conjugate gradient method.
%         u1 = (A' * SIGMA_INV * A + C_INV) \ ((A * SIGMA_INV * y_obs(:)));
%
%
%         % u1 = inv(A' * SIGMA_INV * A + C_INV) * ((A * SIGMA_INV * y_obs(:)));
%         % Solve the minimization problem using conjugate gradient method.
%         RHS = A' * (SIGMA_INV(1, 1) * y_obs(:)); % Construct right hand side
%         x_0 = zeros(size(RHS));
%         matvecc = @(x) A' * (SIGMA_INV(1, 1) * (A * x)) + C_INV(1, 1) * x;
%         max_iters = 500; tol = 1e-8;
%         u1 = CG(matvecc, RHS, x_0, max_iters, tol, true);
%
%         error(n_al) = norm(u1 - true_solution) / norm(true_solution)*100;
%     end
% end
% semilogx(al,error)







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