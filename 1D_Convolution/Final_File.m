clear all; clc; close all;
rand('state',20)

mesh_array = [50000];
for n = 1:numel(mesh_array)
    N = mesh_array(n);
    N_ITER_array= [1 10];    % number of iterations 1/10/100
    r_array=[N/10 N/5 N/2]; % 10 / 20 / 50 percent of mesh size
    
    % --------- Initial parameters for the N mesh size -----------------------
    noise_level = 0.05;
    alpha=2*pi; beta = 0.1;
    [y_synthetic, s] = synthetic_observation(N,alpha,beta,2,N);
    A = linear_operator(N,beta,2,N);
    SIZE_A=size(A); % size of matrix A in R:mxn
    true_solution=conv_on(s,alpha,N);
    y_obs = y_synthetic + max(y_synthetic) * rand(size(y_synthetic)) * noise_level;
    Data_var = noise_level^2; Regularization = 5;
    % -------------------------------------------------------------------------
    
    result_RIGHT = zeros(N,numel(r_array),numel(N_ITER_array));
    result_LEFT = zeros(N,numel(r_array),numel(N_ITER_array));
    result_RAN_MAP = zeros(N,numel(r_array),numel(N_ITER_array));
    
    for iter = 1:length(N_ITER_array)
        N_ITER = N_ITER_array(iter);
        C = 1/Regularization*eye(SIZE_A(2));
        C_INV = Regularization*eye(SIZE_A(2));
        SIGMA = (Data_var)*eye(SIZE_A(1));
        SIGMA_INV = 1/Data_var*eye(SIZE_A(1));
        
        for i = 1:length(r_array)
            r = r_array(i);
            
            result_RIGHT_in_ITER=zeros(N,1);
            result_LEFT_in_ITER=zeros(N,1);
            result_RAN_MAP_in_ITER=zeros(N,1);
            for j = 1:N_ITER
                %% Using rank one product
                EPSILON = normrnd(0,sqrt(1/Regularization),[SIZE_A(2),r]);
                LAMBDA =  normrnd(0,sqrt(1/Data_var),[size(y_obs(:),1),r]);
                sig_rand = normrnd(0,sqrt(Data_var),[size(y_obs(:),1),r]);

                
                
                %% =================== LEFT SKETCHING =====================
                SIGMA_INV_rand = 1/r * LAMBDA * LAMBDA';
                % Solve the minimization problem using conjugate gradient method.
                RHS = A' * (SIGMA_INV_rand*y_obs(:)); % Construct right hand side
                x_0 = zeros(size(RHS));
                matvecc = @(x) A'*(SIGMA_INV_rand*(A*x)) + C_INV * x;
                max_iters = 500; tol = 1e-5;
                result_LEFT_in_ITER = result_LEFT_in_ITER + 1/N_ITER * ...
                    CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                
                %% ============= RANDOMIZED MAP solution ==================
                % Solve the minimization problem using conjugate gradient method.
                RHS = A'*(SIGMA_INV(1,1)*(y_obs(:) + mean(sig_rand,2))) + C_INV(1,1) * mean(EPSILON,2);
                x_0 = zeros(size(RHS));
                matvecc = @(x) A'*(SIGMA_INV(1,1)*(A*x)) + C_INV(1,1) * x;
                max_iters = 500; tol = 1e-5;
                result_RAN_MAP_in_ITER = result_RAN_MAP_in_ITER + 1/N_ITER * ...
                    CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                
                %% ============ RIGHT SKETCHING ===========================
                C_RAND = 1/r * EPSILON*EPSILON';
                % (1) using CG for solvong Y = (SIGMA + A C A')^{-1} d
                % => (2) Then u_RS  = C A' Y
                % (1) -----------------------------------------------------
                RHS = y_obs(:);
                x_0 = zeros(size(RHS));
                matvecc = @(x) SIGMA(1,1) * x + A*(C_RAND * (A'*x));
                max_iters = 500; tol = 1e-5;
                Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
                % (2) -----------------------------------------------------
                u_RS = C_RAND * (A'*(Y));
                result_RIGHT_in_ITER = result_RIGHT_in_ITER + 1/N_ITER * u_RS;
                
                
            end
            
            result_RIGHT(:,i,iter) = result_RIGHT_in_ITER;
            result_LEFT(:,i,iter) = result_LEFT_in_ITER;
            result_RAN_MAP(:,i,iter)  = result_RAN_MAP_in_ITER;
            
        end
        % u1 = inv(A' * SIGMA_INV * A + C_INV) * ((A * SIGMA_INV * y_obs(:)));
        % Solve the minimization problem using conjugate gradient method.
        RHS = A' * (SIGMA_INV(1,1)*y_obs(:)); % Construct right hand side
        x_0 = zeros(size(RHS));
        matvecc = @(x) A'*(SIGMA_INV(1,1)*(A*x)) + C_INV(1,1) * x;
        max_iters = 500; tol = 1e-5;
        u1 = CG(matvecc, RHS, x_0, max_iters, tol, false);
        
        
        % u2 = (C*A')*(inv(SIGMA + A*C*A') * y_obs);
        RHS = y_obs(:); x_0 = zeros(size(RHS));
        matvecc = @(x) SIGMA(1,1) * x + A*(C(1,1) * (A'*x));
        max_iters = 500; tol = 1e-5;
        Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
        % (2) -----------------------------------------------------
        u2 = C(1,1) * (A'*(Y));
    end
    
    
    %% Plot the results
    % -------- RIGHT SKETCHING --------
    figure
    for iter = 1:length(N_ITER_array)
        subplot(1,2,iter)
        for i = 1:length(r_array)
            plot(s,result_RIGHT(:,i,iter)); hold on
        end
        
        plot(s,u2,'-k','Linewidth',1.5)
        legend(['10% mesh, r = ' num2str(r_array(1))],['20% mesh, r = ' num2str(r_array(2))] ,['50% mesh, r = ' num2str(r_array(3))], 'u_2 solution')
        title(['Realizations ' num2str(N_ITER_array(iter))])
        ylim([-5 5])
    end
    sgtitle(['Ensemble Kalman filter as Right Sketching Methods, Mesh = ' num2str(N)])
    savefig(['RIGHT_MESH_SIZE_' num2str(N)])
    
    % -------- LEFT SKETCHING --------
    figure
    for iter = 1:length(N_ITER_array)
        subplot(1,2,iter)
        for i = 1:length(r_array)
            plot(s,result_LEFT(:,i,iter)); hold on
        end
        
        plot(s,u1,'--k','Linewidth',1.5)
        legend(['10% mesh, r = ' num2str(r_array(1))],['20% mesh, r = ' num2str(r_array(2))] ,['50% mesh, r = ' num2str(r_array(3))], 'u_1 solution')
        title(['Realizations ' num2str(N_ITER_array(iter))])
        ylim([-2 2])
    end
    sgtitle(['Left Sketching Methods, Mesh = ' num2str(N)])
    savefig(['LEFT_MESH_SIZE_' num2str(N)])
    
    % -------- RAN - MAP --------
    figure
    for iter = 1:length(N_ITER_array)
        subplot(1,2,iter)
        for i = 1:length(r_array)
            plot(s,result_RAN_MAP(:,i,iter)); hold on
        end
        
        plot(s,u1,'--k','Linewidth',1.5)
        legend(['10% mesh, r = ' num2str(r_array(1))],['20% mesh, r = ' num2str(r_array(2))] ,['50% mesh, r = ' num2str(r_array(3))], 'u_1 solution')
        title(['Realizations ' num2str(N_ITER_array(iter))])
        ylim([-2 2])
    end
    sgtitle(['Randomized MAP Methods, Mesh = ' num2str(N)])
    savefig(['RAN_MAP_MESH_SIZE_' num2str(N)])
    
end



%% EXTRA FUNCTIONS
% ================= Function for CG =======================================
function y = LHS(x,SIGMA_INV,C_INV)
y = A'*(SIGMA_INV*(Ax_y*x)) + C_INV * x;
end

function y = LHS_RIGHT(x,SIGMA,C)
y = SIGMA * x + Ax_y(C * (A'*x));
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

% ================= function for creating data ============================
function[y_observed, s]= synthetic_observation(mesh_points,alpha,beta,kernel,N)

s=linspace(0,1,mesh_points);

m=conv_on(s,alpha,mesh_points);

A=linear_operator(mesh_points,beta,kernel,N);

y_observed=A*m;

end

function[m]= conv_on(s,alpha,mesh_points)
m=zeros(mesh_points,1);
for k =1:length(s)
    m(k)=sin(alpha*s(k))+cos(alpha*s(k));
end
end

function[A]= linear_operator(mesh_points,beta,KERNEL,N)
if KERNEL == 1
    s=linspace(0,1,mesh_points);
    dx = s(2) - s(1);
    
    A=zeros(mesh_points,mesh_points);
    for i=1:mesh_points
        for j=1:mesh_points
            A(i,j)=kernel(s(i),s(j),beta)/mesh_points;
        end
    end
else
    a = 0.1;
    n = mesh_points;
    Dx = 1/n;
    
    %% CONSTRUCT MATRIX MODEL AND COMPUTE DATA _WITH_ INVERSE CRIME
    % Construct normalized discrete point spread function
    nPSF = ceil(a/Dx);          % number points (v+1) Dx > a
    %-- NUMBER OF POINTS THAT ONE OF PSF LYING IN f(x)
    xPSF = [-nPSF:nPSF]*Dx;     % Coordinate of points
    %-- RELATIVE COORDITNATE OF ONE PSF
    PSF  = zeros(size(xPSF));
    %-- PSF
    ind   = abs(xPSF)<a;        % (v+1) Dx > a => v
    %-- INDEX
    PSF(ind) = DC_PSF(xPSF(ind),a);     % psf = (x + a)^2*(x - a)^2;
    % Ca = ( \int_{-a}^{a} [(x + a)^2*(x - a)^2] dx )^{-1)
    Ca   = 1/(Dx*trapz(PSF));
    PSF  = Ca*PSF;
    %-- PSF AT MUTUAL POINTS
    
    % Construct convolution matrix
    A = Dx*DC_convmtx(PSF,n);
    A = sparse(A);
end
end

function[a] =kernel(t,s,beta)
a=(1/sqrt(2*pi*beta^2))*exp(-(t-s)^2/(2*beta^2));
end

function mtx = DC_convmtx(PSF,n)

% Force PSF to be horizontal vector
PSF = PSF(:).';

% Determine the constant m (length of PSF should be 2*m+1)
m = (length(PSF)-1)/2;

% First: construct convolution matrix with the boundary condition assuming
% zero signal outside the boundaries
tmp = conv_mtx(PSF,n);
mtx = tmp(:,(m+1):(end-m));

% Second: correct for the periodic boundary condition
for iii = 1:m
    mtx(1:m,(end-m+1):end) = tmp(1:m,1:m);
    mtx((end-m+1):end,1:m) = tmp((end-m+1):end,(end-m+1):end);
end
end

function psf = DC_PSF(x,a)

% Check that parameter a is in acceptable range
if (a<=0) | (a>=1/2)
    error('Parameter a should satisfy 0<a<1/2')
end

% Evaluate the polynomial
psf = (x + a).^2.*(x - a).^2;

end


function X = conv_mtx(x,nh)
x1 = x(:);
p = length(x1);
X = zeros(p+nh-1,nh);
for k = 1:nh
    X(k:p+(k-1),k) = x1;
end
[pp,qq] = size(x);
if pp < qq
    X = X.';
end
end
