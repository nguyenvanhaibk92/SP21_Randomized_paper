clear all; clc; close all;
global N P measang
% Noise level
noise_level = 0.01;

% List of mesh point
mesh_array = [256];
para_proper = [1e5];
% MESH      64   128
% par       1e4  1e5
for n = 1:numel(mesh_array)
    N = mesh_array(n); Regularization = para_proper(n);
    N_ITER_array= [1 5 20];    % number of iterations 1/10/100
    r_array=[ceil(N^2/10) ceil(N^2/5) ceil(N^2/2)]; % 10 / 20 / 50 percent of mesh size
    
    target = phantom('Modified Shepp-Logan',N);
    angle0  = -90;
    measang = angle0 + [0:(N-1)]/N*180;
    P  = length(radon(target,0));
    NP = N*P; % size of y_obs
    NN = N^2; % size of x / original image
    observation = radon(target,measang);
    
    % Add noise to the data
    e = randn(size(observation));
    y_obs  = observation + noise_level*max(max(observation))*e;
    Data_var = (noise_level)^2; % covariance matrix for data  \Sigma^{-1} = noise_level
    
    C = 1/Regularization; %*eye(NN);
    SIGMA = (Data_var); %*eye(NP);
    C_INV = Regularization; %*eye(NN);
    SIGMA_INV=(1/Data_var); %*eye(NP);
    
    %================ Initialized save results ============================
    result_RIGHT = zeros(N^2,numel(r_array),numel(N_ITER_array));
    result_LEFT = zeros(N^2,numel(r_array),numel(N_ITER_array));
    result_RAN_MAP = zeros(N^2,numel(r_array),numel(N_ITER_array));
    %----------------------------------------------------------------------
    for iter = 1:length(N_ITER_array)
        N_ITER = N_ITER_array(iter);
        
        for i = 1:numel(r_array)
            r = r_array(i);
            
            result_RIGHT_in_ITER=zeros(NN,1);
            result_LEFT_in_ITER=zeros(NN,1);
            result_RAN_MAP_in_ITER=zeros(NN,1);
            
            for iter_loop = 1:N_ITER
                
                %% =================== LEFT SKETCHING =====================
                LAMBDA = 1/sqrt(r) * normrnd(0,sqrt(1/Data_var),[NP,r]);
                %                 SIGMA_INV_RAND = 1/r * LAMBDA*LAMBDA';
                
                % Solve the minimization problem using conjugate gradient method.
                RHS = ATy_x(LAMBDA*(LAMBDA'*y_obs(:))); % Construct right hand side
                x_0 = zeros(size(RHS));
                %                 matvecc = @(x) LHS(x,SIGMA_INV_RAND,C_INV(1,1)); % Construct left hand side
                matvecc = @(x) LHS(x,LAMBDA,C_INV(1,1));
                max_iters = 500; tol = 1e-5;
                
                result_LEFT_in_ITER = result_LEFT_in_ITER + 1/N_ITER * ...
                    CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                %% ============= RANDOMIZED MAP solution ==================
                EPSILON = normrnd(0,sqrt(1/Regularization),[NN,r]);
                sig_rand = normrnd(0,sqrt(Data_var),[NP,r]);
                
                % Solve the minimization problem using conjugate gradient method.
                RHS = ATy_x(SIGMA_INV(1,1)*(y_obs(:) + mean(sig_rand,2))) + C_INV(1,1) * mean(EPSILON,2); % Construct right hand side
                x_0 = zeros(size(RHS));
                matvecc = @(x) LHS(x,SIGMA_INV(1,1),C_INV(1,1)); % Construct left hand side
                max_iters = 500; tol = 1e-5;
                
                
                result_RAN_MAP_in_ITER = result_RAN_MAP_in_ITER + 1/N_ITER * ...
                    CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                %% ============ RIGHT SKETCHING ===========================
                EPSILON = 1/sqrt(r) * normrnd(0,sqrt(1/Regularization),[NN,r]);
%                 C_RAND = EPSILON*EPSILON';
                
                % (1) using CG for solvong Y = (SIGMA + A C A')^{-1} d
                % => (2) Then u_RS  = C A' Y
                % (1) ---------------------
                RHS = y_obs(:);
                x_0 = zeros(size(RHS));
                matvecc = @(x) LHS_RIGHT(x,SIGMA(1,1),EPSILON); % Construct left hand side
                max_iters = 500; tol = 1e-5;
                Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                % (2) ---------------------
                u_RS = EPSILON*(EPSILON' * ATy_x(Y));
                result_RIGHT_in_ITER = result_RIGHT_in_ITER + 1/N_ITER * u_RS;
                xxx
            end
            result_LEFT(:,i,iter) = result_LEFT_in_ITER;
            result_RAN_MAP(:,i,iter) = result_RAN_MAP_in_ITER;
            result_RIGHT(:,i,iter) = result_RIGHT_in_ITER;
            
        end
    end
    
    %% ========================= u_1 solution =============================
    % Solve the minimization problem using conjugate gradient method.
    RHS = ATy_x(SIGMA_INV(1,1)*y_obs(:)); % Construct right hand side
    x_0 = zeros(size(RHS));
    matvecc = @(x) LHS(x,sqrt(SIGMA_INV(1,1)),C_INV(1,1)); % Construct left hand side
    max_iters = 500; tol = 1e-5;
    u_1 = CG(matvecc, RHS, x_0, max_iters, tol, false);
    
    %% ========================= u_2 solution =============================
    % Solve the minimization problem using conjugate gradient method.
    % (1) using CG for solvong Y = (SIGMA + A C A')^{-1} d
    % => (2) Then u_RS  = C A' Y
    % step (1) ---------------------
    RHS = y_obs(:); x_0 = zeros(size(RHS));
    matvecc = @(x) LHS_RIGHT(x,SIGMA(1,1),sqrt(C(1,1))); % Construct left hand side
    max_iters = 500; tol = 1e-5;
    Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
    
    % step (2) ---------------------
    u_2 = C(1,1) * ATy_x(Y);
    % ---------------------------------------------------------------------
    
    %% PLOT RESULTS
    % ======================== LEFT SKETCHING =============================
    figure
    for iter = 1:length(N_ITER_array)
        subplot(numel(N_ITER_array),numel(r_array)+1,(numel(r_array)+1)*(iter-1)+1)
        imagesc(reshape(u_1,[N,N]))
        axis square; colormap gray; axis off;
        text(-20,35,['Iter = ' num2str(N_ITER_array(iter))])
        title('u_1 solution')
        
        for i = 1:length(r_array)
            subplot(numel(N_ITER_array),numel(r_array)+1,(numel(r_array)+1)*(iter-1)+i+1)
            imagesc(reshape(result_LEFT(:,i,iter),[N,N]))
            axis square; axis off; colormap gray;
            error = norm(u_1-result_LEFT(:,i,iter))/norm(u_1);
            title(['r = ' num2str(round(r_array(i)/N^2*100,0)) '%   Error ' num2str(round(error*100,1)) '%'])
        end
    end
    sgtitle(['LEFT SKETCHING, ' num2str(N) 'x' num2str(N) ' Pixels'])
    savefig(['LEFT_SKETCHING_MESH_SIZE_' num2str(N) 'x' num2str(N)])
    % ======================== RAN MAP ====================================
    figure
    for iter = 1:length(N_ITER_array)
        subplot(numel(N_ITER_array),(numel(r_array)+1),(numel(r_array)+1)*(iter-1)+1)
        imagesc(reshape(u_1,[N,N]))
        axis square; colormap gray; axis off;
        text(-20,35,['Iter = ' num2str(N_ITER_array(iter))])
        title('u_1 solution')
        for i = 1:length(r_array)
            subplot(numel(N_ITER_array),(numel(r_array)+1),(numel(r_array)+1)*(iter-1)+i+1)
            imagesc(reshape(result_RAN_MAP(:,i,iter),[N,N]))
            axis square; axis off; colormap gray;
            error = norm(u_1-result_RAN_MAP(:,i,iter))/norm(u_1);
            title(['r = ' num2str(round(r_array(i)/N^2*100,0)) '%   Error ' num2str(round(error*100,1)) '%'])
        end
    end
    sgtitle(['RAN MAP, ' num2str(N) 'x' num2str(N) ' Pixels'])
    savefig(['RAN_MAP_MESH_SIZE_' num2str(N) 'x' num2str(N)])
    
    % ======================== RIGHT SKETCHING ============================
    figure
    for iter = 1:length(N_ITER_array)
        subplot(numel(N_ITER_array),(numel(r_array)+1),(numel(r_array)+1)*(iter-1)+1)
        imagesc(reshape(u_2,[N,N]))
        axis square; colormap gray; axis off;
        text(-20,35,['Iter = ' num2str(N_ITER_array(iter))])
        title('u_2 solution')
        for i = 1:length(r_array)
            subplot(numel(N_ITER_array),(numel(r_array)+1),(numel(r_array)+1)*(iter-1)+i+1)
            imagesc(reshape(result_RIGHT(:,i,iter),[N,N]))
            axis square; axis off; colormap gray;
            error = norm(u_2-result_RIGHT(:,i,iter))/norm(u_2);
            title(['r = ' num2str(round(r_array(i)/N^2*100,0)) '%   Error ' num2str(round(error*100,1)) '%'])
        end
    end
    sgtitle(['RIGHT SKETCHING, ' num2str(N) 'x' num2str(N) ' Pixels'])
    savefig(['RIGHT_SKETCHING_MESH_SIZE_' num2str(N) 'x' num2str(N)])
end









%% EXTRA FUNCTIONS
% Action of matrix A is randon()
% 40.73499999*N/64; Incomprehensible correction factor
% Action of matrix A^{-1} is irandon()*corxn


function y = LHS(x,LAMBDA,C_INV)
y = (ATy_x(LAMBDA*(LAMBDA'*(Ax_y(x))))) + C_INV * x;
end

function y = LHS_RIGHT(x,SIGMA,EPSILON)
% y = (SIGMA + A C A')
y = SIGMA * x + Ax_y(EPSILON* (EPSILON' * ATy_x(x)));
end

function [y] = Ax_y(x) % operator A*x
global N measang

y = radon(reshape(x,[N,N]),measang);
y = y(:); % return column size [P*N,1]
end


function x = ATy_x(y)   % operator A'*x
global N P measang

x = 40.73499999*N/64 * iradon(reshape(y,[P,N]), measang, 'none');
x = x(2:end-1, 2:end-1);
x = x(:); % return column size [N*N,1]
end


function y = CG(matvec, RHS, x_0, max_iters, tol, should_print)
y      = x_0;
r      = RHS; % RHS - matvec(0) = RHS since matvec is linear
r_0    = r;
norm_r = norm(r_0,2);
if tol == 0
    tol = min(0.5, norm_r) * norm(r);
else
    tol = tol * norm(r);
end
rho    = r'*r;
i      = 1;
while i <= max_iters && norm(r,2) > tol
    if i==1
        p = r;
    else
        beta  = rho/old_rho;
        old_p = p;
        p     = r + beta*p;
    end
    w          = matvec(p);
    den        = (p'*w);
    a          = rho/den;
    y_old      = y;
    y          = y + a*p;
    r          = r - a*w;
    old_rho    = rho;
    rho        = r'*r;
    
    %     if should_print && mod(i,5) == 0
    %         fprintf('Iteration %4d of %4d \n',i,max_iters)
    %         fprintf('r norm: %0.4e \n', norm(r,2))
    %     elseif mod(i,100)==0
    %         fprintf('%4d %4d  %0.6e\n',i, max_iters, norm(r,2))
    %     end
    
    i = i+1;
end
end