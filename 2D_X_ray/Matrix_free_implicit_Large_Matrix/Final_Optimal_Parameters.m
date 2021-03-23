clear all; clc; close all;
global N P measang
% Noise level
noise_level = 0.01;

% List of mesh point
mesh_points = [512];

% List of regularization parameters
parameters = logspace(-4,10,100);

% Number of iterations and sampling
N_ITER = 1;
r = 1;

for n = 1:length(mesh_points)
    N = mesh_points(n);
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
    
    for i = 1:length(parameters)
        i
        Regularization = parameters(i);
        C = 1/Regularization*eye(1);
        SIGMA = (Data_var)*eye(1);
        C_INV = Regularization*eye(1);
        
        SIGMA_INV=(1/Data_var)*eye(1);
        L_SIG_INV=(1/sqrt(Data_var))*ones(NP,1);
        
        % Construct right hand side
        RHS = ATy_x(SIGMA_INV(1,1)*y_obs(:));
        
        % Solve the minimization problem using conjugate gradient method.
        % See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
        x_0 = zeros(size(RHS));
        matvecc = @(x) LHS(x,SIGMA_INV,C_INV);
        max_iters = 500; tol = 1e-5;
        result_MAP1 = CG(matvecc, RHS, x_0, max_iters, tol, false);
        eta_MAP1(i,1) = norm(y_obs(:) - Ax_y(result_MAP1));
        rho_Map1(i,1) = norm(result_MAP1);
        Error_Map1(i,1) = norm(target(:)-result_MAP1)/norm(target(:))*100;
        
    end
end
%% Parameters L-Curve and Errors

for nn = 1:length(mesh_points)
    figure
    subplot(2,1,1)
    semilogx(eta_MAP1(:,nn),rho_Map1(:,nn),'--pr'); hold on
    legend('MAP1')
    xlabel('Residual norm ||b-Ax||')
    ylabel('norm of solution ||x||')
    title('Optimal parameter')
    subplot(2,1,2)
    semilogx(parameters,Error_Map1(:,nn),'--pr'); hold on
    legend('MAP1')
    xlabel('Regularization \alpha')
    ylabel('Error % ||x_{true}-x_{rec}|| / ||x_{true}||')
    title('Error versus \alpha')
    sgtitle(['Mesh size = ' num2str(N)])
end

savefig(['Optimal parameter, Mesh size = ' num2str(N) 'x' num2str(N)])



clear all; clc; close all;
global N P measang
% Noise level
noise_level = 0.01;

% List of mesh point
<<<<<<< HEAD
mesh_points = [512];
=======
mesh_points = [1024];
>>>>>>> d215a3da2217b40f607d0c589924fc1f30741d83

% List of regularization parameters
parameters = logspace(-4,10,100);

% Number of iterations and sampling
N_ITER = 1;
r = 1;

for n = 1:length(mesh_points)
    N = mesh_points(n);
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
    
    for i = 1:length(parameters)
        i
        Regularization = parameters(i);
        C = 1/Regularization*eye(1);
        SIGMA = (Data_var)*eye(1);
        C_INV = Regularization*eye(1);
        
        SIGMA_INV=(1/Data_var)*eye(1);
        L_SIG_INV=(1/sqrt(Data_var))*ones(NP,1);
        
        % Construct right hand side
        RHS = ATy_x(SIGMA_INV(1,1)*y_obs(:));
        
        % Solve the minimization problem using conjugate gradient method.
        % See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
        x_0 = zeros(size(RHS));
        matvecc = @(x) LHS(x,SIGMA_INV,C_INV);
        max_iters = 500; tol = 1e-5;
        result_MAP1 = CG(matvecc, RHS, x_0, max_iters, tol, false);
        eta_MAP1(i,1) = norm(y_obs(:) - Ax_y(result_MAP1));
        rho_Map1(i,1) = norm(result_MAP1);
        Error_Map1(i,1) = norm(target(:)-result_MAP1)/norm(target(:))*100;
        
    end
end
%% Parameters L-Curve and Errors

for nn = 1:length(mesh_points)
    figure
    subplot(2,1,1)
    semilogx(eta_MAP1(:,nn),rho_Map1(:,nn),'--pr'); hold on
    legend('MAP1')
    xlabel('Residual norm ||b-Ax||')
    ylabel('norm of solution ||x||')
    title('Optimal parameter')
    subplot(2,1,2)
    semilogx(parameters,Error_Map1(:,nn),'--pr'); hold on
    legend('MAP1')
    xlabel('Regularization \alpha')
    ylabel('Error % ||x_{true}-x_{rec}|| / ||x_{true}||')
    title('Error versus \alpha')
    sgtitle(['Mesh size = ' num2str(N)])
end

savefig(['Optimal parameter, Mesh size = ' num2str(N) 'x' num2str(N)])
































































































%% EXTRA FUNCTIONS
% Action of matrix A is randon()
% 40.73499999*N/64; Incomprehensible correction factor
% Action of matrix A^{-1} is irandon()*corxn


function y = LHS(x,SIGMA_INV,C_INV)
y = (ATy_x(SIGMA_INV(1,1)*(Ax_y(x)))) + C_INV(1,1) * x;
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
    
    if should_print && mod(i,5) == 0
        fprintf('Iteration %4d of %4d \n',i,max_iters)
        fprintf('r norm: %0.4e \n', norm(r,2))
    elseif mod(i,100)==0
        fprintf('%4d %4d  %0.6e\n',i, max_iters, norm(r,2))
    end
    
    i = i+1;
end
end

