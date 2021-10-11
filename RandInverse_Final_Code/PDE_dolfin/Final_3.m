clear all; clc; close all;
rand('state', 20)
mesh_array = [2868];

N = mesh_array(1);

% --------- Initial parameters for the N mesh size ------------------------
noise_level = 0.01;
load('observation.mat')
y_obs = x';
y_obs = y_obs + max(y_obs) * rand(size(y_obs)) * noise_level;

load('linear_operator.mat')
A = A; % matrix A, Operator
SIZE_A = size(A); % size of matrix A in R:mxn

load('initial_condition.mat')
u0 = 0.25 * ones(size(A'*y_obs(:)));
true_solution = x';

load('Regularization.mat')
for i = 1:N
    for j = 1:N
        X(j,i) = x(i,j);
    end
end
C_INV = X;
C = inv(C_INV);

for i = 1:N
    for j = 1:N
        C(j,i) = C(i,j);
    end
end

Data_var = (max(y_obs)*noise_level)^2;

% C = 1 / Regularization * eye(SIZE_A(2));
% C_INV = Regularization * eye(SIZE_A(2));
SIGMA = (Data_var) * eye(SIZE_A(1));
SIGMA_INV = 1 / Data_var * eye(SIZE_A(1));

max_iters = 1000; tol = 1e-10;

% u1 = inv(A' * SIGMA_INV * A + C_INV) * ((A * SIGMA_INV * y_obs(:)));
RHS = A' * (SIGMA_INV(1,1) * y_obs(:)) + C_INV * u0;
x_0 = zeros(size(RHS));
matvecc = @(x) A' * (SIGMA_INV(1,1) * (A * x)) + C_INV * x;
u1 = CG(matvecc, RHS, x_0, max_iters, tol, false);

% -------------------------------------------------------------------------
r_array = ceil([250 750 1500]); % 10 / 20 / 50 percent of mesh size
M = [750 1500];
realization = [1 20];
r_svd = [1 20 100]; % max size is 200 observations


result_rMAP = zeros(N,numel(realization));
result_LEFT = zeros(N,numel(r_array),numel(realization));
result_RMA = zeros(N,numel(r_array),numel(realization));
result_RIGHT = zeros(N,numel(r_array),numel(realization));
result_EnKF = zeros(N,numel(r_array),numel(realization));
result_rSVD = zeros(N,numel(r_svd));


for re = 1:numel(realization)
    % Using rank one product
    EPSILON = mvnrnd(zeros(SIZE_A(2),1),C, M(re))'; % draw (0,C)
    LAMBDA = normrnd(0, sqrt(1 / Data_var), [size(y_obs(:), 1), M(re)]);% draw (0,Sigma^{-1})
    sig_rand = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), M(re)]);  % draw (0,Sigma)
    
    % rMAP
    RHS = A' * (SIGMA_INV(1, 1) * (y_obs(:) + mean(sig_rand, 2))) + C_INV * (u0 + mean(EPSILON, 2));
    x_0 = zeros(size(RHS));
    matvecc = @(x) A' * (SIGMA_INV(1, 1) * (A * x)) + C_INV * x;
    result_rMAP(:,re) = CG(matvecc, RHS, x_0, max_iters, tol, false);
end

for realize = 1:numel(realization)
    n_realize = realization(realize);
    for jj = 1:n_realize
        for i = 1:length(r_array)
            r = r_array(i);
            [n_realize jj r]
            
            % Using rank one product
            EPSILON = mvnrnd(zeros(SIZE_A(2),1),C, M(re))'; % draw (0,C)
            % delta_rand = mvnrnd(zeros(SIZE_A(2),1),C_INV, M(re))';  % draw (0,C^{-1})
            LAMBDA = normrnd(0, sqrt(1 / Data_var), [size(y_obs(:), 1), r]);% draw (0,Sigma^{-1})
            sig_rand = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), r]);  % draw (0,Sigma)
            
            
            % LEFT
            SIGMA_INV_rand = 1 / r * LAMBDA * LAMBDA';
            % Solve the minimization problem using conjugate gradient method.
            RHS = A' * (SIGMA_INV_rand * y_obs(:)) + C_INV * u0; % Construct right hand side
            x_0 = zeros(size(RHS));
            matvecc = @(x) A' * (SIGMA_INV_rand * (A * x)) + C_INV * x;
            result_LEFT(:,i,realize) = result_LEFT(:,i,realize) + 1/n_realize *...
                CG(matvecc, RHS, x_0, max_iters, tol, false);
            
            
            % RAM
            re = realize;
            
            % randomize the whole matrix to take average
            Epsilon = mvnrnd(zeros(SIZE_A(2),1),C, M(re))'; % draw (0,C)
            Sigma_rand = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), M(re)]);  % draw (0,Sigma)
            % Solve the minimization problem using conjugate gradient method.
            RHS = A' * (SIGMA_INV_rand * (y_obs(:) + mean(Sigma_rand, 2))) + C_INV * (u0 + mean(EPSILON, 2));
            x_0 = zeros(size(RHS));
            matvecc = @(x) A' * (SIGMA_INV_rand * (A * x)) + C_INV * x;
            u_LS = CG(matvecc, RHS, x_0, max_iters, tol, false);
            
            result_RMA(:,i,realize) = result_RMA(:,i,realize) + 1/n_realize * u_LS;
            
            % RIGHT
            C_RAND = 1 / r * EPSILON * EPSILON';
            % (1) using CG for solvong Y = (SIGMA + A C A')^{-1} d
            % => (2) Then u_RS  = C A' Y
            % (1) -----------------------------------------------------
            RHS = y_obs(:) - A * u0;
            x_0 = zeros(size(RHS));
            matvecc = @(x) SIGMA(1, 1) * x + A * (C_RAND * (A' * x));
            Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
            % (2) -----------------------------------------------------
            u_RS = u0 +  C_RAND * (A' * (Y));
            result_RIGHT(:,i,realize) = result_RIGHT(:,i,realize) + 1/n_realize * u_RS;
            
            
            % EnKF
            result_RAM_in_ITER = zeros(N, 1);
            for j = 1:M(re)
                % EnKF right sketching
                delta = mvnrnd(zeros(SIZE_A(2),1),C, 1)';
                sigma = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), 1]);
                % (1) -----------------------------------------------------
                RHS = (y_obs(:) + sigma - A *(u0 + delta));
                x_0 = zeros(size(RHS));
                matvecc = @(x) SIGMA(1, 1) * x + A * (C_RAND * (A' * x));
                Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
                % (2) -----------------------------------------------------
                u_RS = u0 + delta + C_RAND * (A' * (Y));
                result_RAM_in_ITER = result_RAM_in_ITER + 1/M(re) * u_RS;
            end
            result_EnKF(:,i,realize) = result_EnKF(:,i,realize) + 1/n_realize * result_RAM_in_ITER;
            
            norm(result_rMAP(:,re) - u1) / norm(u1)
            norm(result_LEFT(:,i,re) - u1) / norm(u1)
            norm(result_RMA(:,i,re) - u1) / norm(u1)
            norm(result_RIGHT(:,i,re) - u1) / norm(u1)
            norm(result_EnKF(:,i,re) - u1) / norm(u1)
            
        end
    end
end

save('result_solutions','result_rMAP','result_LEFT','result_RMA','result_RIGHT','result_EnKF')

for re = 1:numel(realization)
    error(re,1) = norm(result_rMAP(:,re) - u1) / norm(u1);
    for i = 1:length(r_array)
        error((re-1)*3+i,2) = norm(result_LEFT(:,i,re) - u1) / norm(u1);
        error((re-1)*3+i,3) = norm(result_RMA(:,i,re) - u1) / norm(u1);
        error((re-1)*3+i,4) = norm(result_RIGHT(:,i,re) - u1) / norm(u1);
        error((re-1)*3+i,5) = norm(result_EnKF(:,i,re) - u1) / norm(u1);
    end
end

for re = 1:numel(realization)
    result = result_rMAP(:,re);
    save(['INITIAL_CONDITION_rMAP_' num2str(re)],'result')
    for i = 1:length(r_array)
        result = result_LEFT(:,i,re);
        save(['INITIAL_CONDITION_LEFT_' num2str((re-1)*3+i)],'result')
        result = result_RMA(:,i,re);
        save(['INITIAL_CONDITION_RAM_' num2str((re-1)*3+i)],'result')
        result = result_RIGHT(:,i,re);
        save(['INITIAL_CONDITION_RIGHT_' num2str((re-1)*3+i)],'result')
        result = result_EnKF(:,i,re);
        save(['INITIAL_CONDITION_EnKF_' num2str((re-1)*3+i)],'result')
    end
end








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


function [U,S,V] = rsvd(A,r,q,p)
% Step 1: Sample column space of A with P matrix
n = size(A,2);
P = randn(n,r + p);
Y = A * P;
% Randomized Power Iteration Algorithm 4.3
for k = 1:q
    Y = A * (A' * Y);
end
[Q,R] = qr(Y,0);

% Step 2: Compute SVD on projected Y = Q'*A;
B = Q' * A;
[UB,S,V] = svd(B,'econ');
U = Q * UB;
end

