clear all; clc; close all;
rand('state', 20)
mesh_array = [200];

for problem = 7
    N = mesh_array(1);
    
    if problem == 7
        % --------- Initial parameters for the N mesh size -----------------------
        noise_level = 0.05;
        alpha = 2 * pi; beta = 0.1;
        [y_synthetic, s] = synthetic_observation(N, alpha, beta, 2, N);
        A = linear_operator(N, beta, 2, N);
        SIZE_A = size(A); % size of matrix A in R:mxn
        true_solution = conv_on(s, alpha, N);
        y_obs = y_synthetic + max(y_synthetic) * rand(size(y_synthetic)) * noise_level;
        Data_var = noise_level^2; Regularization = 5;
        name = 'Decon';
        
        factor = 5; % scale graph
        M7 = [N/4 N/2];
        reg_opt(problem) = Data_var * Regularization;  % reg_opt(problem)  = (0.0689261210434970^2);
    else
        n = N; % number of grid points
        noise = 1; % noise level is 5% for all problem
        s = linspace(0,1,n); % x axis
        sp = floor(n/10);% for plot markers
        reg_opt = [0.0135304777457981,0.138488637139387,0.000657933224657568,0.00335160265093884,0.174752840000768,0.0170735264747069,0.0689261210434970];
        
        if problem == 1
            [A,y0,xtrue] = shaw(n);
            name = 'Shaw';
        elseif problem == 2
            [A,y0,xtrue] = gravity(n);
            name = 'Gravity';
        elseif problem == 3
            [A,y0,xtrue] = deriv2(n);
            name = 'Deriv2';
        elseif problem == 4
            [A,y0,xtrue] = heat(n);
            name = 'Heat';
        elseif problem == 5
            [A,y0,xtrue] = phillips(n);
            name = 'Phillips';
        elseif problem == 6
            [A,y0,xtrue] = foxgood(n);
            name = 'Foxgood';
        end
        true_solution = xtrue;
        
        STD_noise = max(abs(y0))*noise/100;
        y_obs = y0 + STD_noise*randn(size(y0));
        
        Data_var = STD_noise^2;
        Regularization = 1/(Data_var/reg_opt(problem));
        
        SIZE_A = size(A); % size of matrix A in R:mxn
        
        factor = 1.1; % scale graph
        M7 = [N/4 N/2];
    end
    
    % -------------------------------------------------------------------------
    C = 1 / Regularization * eye(SIZE_A(2));
    C_INV = Regularization * eye(SIZE_A(2));
    SIGMA = (Data_var) * eye(SIZE_A(1));
    SIGMA_INV = 1 / Data_var * eye(SIZE_A(1));
    
    max_iters = 500; tol = 1e-5;  % Solve the minimization problem using conjugate gradient method.
    
    % u1 = inv(A' * SIGMA_INV * A + C_INV) * ((A * SIGMA_INV * y_obs(:)));
    RHS = A' * (SIGMA_INV(1, 1) * y_obs(:)); % Construct right hand side
    x_0 = zeros(size(RHS));
    matvecc = @(x) A' * (SIGMA_INV(1, 1) * (A * x)) + C_INV(1, 1) * x;
    u1 = CG(matvecc, RHS, x_0, max_iters, tol, false);
    
    % u2 = (C*A')*(inv(SIGMA + A*C*A') * y_obs);
    RHS = y_obs(:); x_0 = zeros(size(RHS));
    matvecc = @(x) SIGMA(1, 1) * x + A * (C(1, 1) * (A' * x));
    Y = CG(matvecc, RHS, x_0, max_iters, tol, false);
    u2 = C(1, 1) * (A' * (Y));
    
    % ---------------------------------------------------------------------
    r_array = [N/10 N/5 N/2]; % 10 / 20 / 50 percent of mesh size
    r_svd = [1 10 100];
    M = [N/4 N/2];
    realization = [1 10];
    
    result_rMAP = zeros(N,numel(realization));
    result_LEFT = zeros(N,numel(r_array),numel(realization));
    result_RMA = zeros(N,numel(r_array),numel(realization));
    result_RIGHT = zeros(N,numel(r_array),numel(realization));
    result_EnKF = zeros(N,numel(r_array),numel(realization));
    result_rSVD = zeros(N,numel(r_svd));
    
    
    color(1,:) = [0 0.4470 0.7410];
    color(2,:) = [0.6350 0.0780 0.1840];
    color(3,:) = [0.4660 0.6740 0.1880];
    
    
    for r_appr = 1:numel(r_svd)
        %% Randomized svd method
        k = r_svd(r_appr);     % Target rank
        q = 1;      % Power iteration to deal with singular spectrum decays slowly
        p = 5;      % Oversampling parameter; p = 5 or 10 gives superb results
        [U,D,V] = rsvd(A,k,q,p);
        result_rSVD(:,r_appr) = V * (D.^2 + reg_opt(problem)*eye(k+p))^-1 * D * U'*y_obs;
        
    end
    %% Randomized methods
    for re = 1:numel(realization)
        % Using rank one product
        EPSILON = normrnd(0, sqrt(1 / Regularization), [SIZE_A(2), M(re)]); % draw (0,C)
        delta_rand = normrnd(0, sqrt(Regularization), [SIZE_A(2), M(re)]);  % draw (0,C^{-1})
        LAMBDA = normrnd(0, sqrt(1 / Data_var), [size(y_obs(:), 1), M(re)]);% draw (0,Sigma^{-1})
        sig_rand = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), M(re)]);  % draw (0,Sigma)
        
        % Randomized MAP
        % Solving
        RHS = A' * (SIGMA_INV(1, 1) * (y_obs(:) + mean(sig_rand, 2))) + C_INV(1, 1) * mean(EPSILON, 2);
        x_0 = zeros(size(RHS));
        matvecc = @(x) A' * (SIGMA_INV(1, 1) * (A * x)) + C_INV(1, 1) * x;
        
        result_rMAP(:,re) = CG(matvecc, RHS, x_0, max_iters, tol, false);
    end
    
    for realize = 1:numel(realization)
        n_realize = realization(realize);
        for jj = 1:n_realize
            for i = 1:length(r_array)
                r = r_array(i);
                
                
                % Using rank one product
                EPSILON = normrnd(0, sqrt(1 / Regularization), [SIZE_A(2), r]); % draw (0,C)
                delta_rand = normrnd(0, sqrt(Regularization), [SIZE_A(2), r]);  % draw (0,C^{-1})
                LAMBDA = normrnd(0, sqrt(1 / Data_var), [size(y_obs(:), 1), r]);% draw (0,Sigma^{-1})
                sig_rand = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), r]);  % draw (0,Sigma)
                
                
                % LEFT
                SIGMA_INV_rand = 1 / r * LAMBDA * LAMBDA';
                % Solve the minimization problem using conjugate gradient method.
                RHS = A' * (SIGMA_INV_rand * y_obs(:)); % Construct right hand side
                x_0 = zeros(size(RHS));
                matvecc = @(x) A' * (SIGMA_INV_rand * (A * x)) + C_INV * x;
                max_iters = 10; tol = 1e-5;
                result_LEFT(:,i,realize) = result_LEFT(:,i,realize) + 1/n_realize *...
                    CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                
                % RAM
                re = realize;
                
                [n_realize jj re r]
                
                % randomize the whole matrix to take average
                Epsilon = normrnd(0, sqrt(1 / Regularization), [SIZE_A(2), M(re)]); % draw (0,C)
                Sigma_rand = normrnd(0, sqrt(Data_var), [size(y_obs(:), 1), M(re)]);  % draw (0,Sigma)
                % Solve the minimization problem using conjugate gradient method.
                RHS = A' * (SIGMA_INV_rand * (y_obs(:) + mean(Sigma_rand, 2))) + C_INV(1, 1) * mean(Epsilon, 2);
                x_0 = zeros(size(RHS));
                matvecc = @(x) A' * (SIGMA_INV_rand * (A * x)) + C_INV * x;
                max_iters = 10; tol = 1e-5;
                u_LS = CG(matvecc, RHS, x_0, max_iters, tol, false);
                
                result_RMA(:,i,realize) = result_RMA(:,i,realize) + 1/n_realize * u_LS;
                
            end
        end
    end
    
    
    figure
    plot(s, u1, '--k', 'Linewidth', 1,'DisplayName','u_1 solution'); hold on
    for r_appr = 1:numel(r_svd)
        plot(s, result_rSVD(:, r_appr),'color',color(r_appr,:), 'Linewidth', 1,'DisplayName',['r = ' num2str(r_svd(r_appr))]);
        error(r_appr,6,problem) = norm(result_rSVD(:, r_appr) - u1)/norm(u1);
        legend('Location','southeast')
        ylim([-1.5*max(abs(true_solution)) 1.5*max(abs(true_solution))])
    end
    set(findall(gcf,'-property','FontSize'),'FontSize',12,'FontName', 'Times New Roman')
    saveas(gcf,['1D_' name '_rSVD'],'epsc')
    
    % rMap
    for re = 1:numel(realization) % take the second relization value to be fair
        n_realize = realization(re);
        figure
        plot(s, u1, '--k', 'Linewidth', 1,'DisplayName','u_1 solution'); hold on
        plot(s, result_rMAP(:, re),'color',color(re,:), 'Linewidth', 1,'DisplayName',['N = ' num2str(M(re))]);
        error((realize-1)*3+re,1,problem) = norm(result_rMAP(:, re) - u1)/norm(u1);
        legend('Location','southeast')
        ylim([-1.1*max(abs(true_solution)) 1.1*max(abs(true_solution))])
        set(findall(gcf,'-property','FontSize'),'FontSize',12,'FontName', 'Times New Roman')
        saveas(gcf,['1D_' name '_rMAP_realization_' num2str(n_realize)],'epsc')
    end
    
    % Left
    for realize = 1:numel(realization)
        n_realize = realization(realize);
        figure
        plot(s, u1, '--k', 'Linewidth', 1,'DisplayName','u_1 solution'); hold on
        for i = 1:length(r_array)
            plot(s, result_LEFT(:, i,realize),'color',color(i,:), 'Linewidth', 1,'DisplayName',['r = ' num2str(r_array(i))]);
            error((realize-1)*3+i,2,problem) = norm(result_LEFT(:, i,realize) - u1)/norm(u1);
            legend('Location','southeast')
        end
        ylim([-1.1*max(abs(true_solution)) 1.1*max(abs(true_solution))])
        set(findall(gcf,'-property','FontSize'),'FontSize',12,'FontName', 'Times New Roman')
        saveas(gcf,['1D_' name '_LS_realization_' num2str(n_realize)],'epsc')
    end
    
    % RAM
    for realize = 1:numel(realization)
        n_realize = realization(realize);
        figure
        plot(s, u1, '--k', 'Linewidth', 1,'DisplayName','u_1 solution'); hold on
        for i = 1:length(r_array)
            plot(s, result_RMA(:, i,realize),'color',color(i,:), 'Linewidth', 1,'DisplayName',['r = ' num2str(r_array(i))]);
            error((realize-1)*3+i,3,problem) = norm(result_RMA(:, i,realize) - u1)/norm(u1);
            legend('Location','southeast')
        end
        ylim([-1.1*max(abs(true_solution)) 1.1*max(abs(true_solution))])
        set(findall(gcf,'-property','FontSize'),'FontSize',12,'FontName', 'Times New Roman')
        saveas(gcf,['1D_' name '_RAM_realization_' num2str(n_realize)],'epsc')
    end
    
end

% save('result','error')
















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
    
    %     if should_print && mod(i,5) == 0
    %         fprintf('Iteration %4d of %4d \n',i,max_iters)
    %         fprintf('r norm: %0.4e \n', norm(r,2))
    %     elseif mod(i,100)==0
    %         fprintf('%4d %4d  %0.6e\n',i, max_iters, norm(r,2))
    %     end
    
    i = i + 1;
end

end

% ================= function for creating data ============================
function [y_observed, s] = synthetic_observation(mesh_points, alpha, beta, kernel, N)

s = linspace(0, 1, mesh_points);

m = conv_on(s, alpha, mesh_points);

A = linear_operator(mesh_points, beta, kernel, N);

y_observed = A * m;

end

function [m] = conv_on(s, alpha, mesh_points)
m = zeros(mesh_points, 1);

for k = 1:length(s)
    m(k) = sin(alpha * s(k)) + cos(alpha * s(k));
end

end

function [A] = linear_operator(mesh_points, beta, KERNEL, N)

if KERNEL == 1
    s = linspace(0, 1, mesh_points);
    dx = s(2) - s(1);
    
    A = zeros(mesh_points, mesh_points);
    
    for i = 1:mesh_points
        
        for j = 1:mesh_points
            A(i, j) = kernel(s(i), s(j), beta) / mesh_points;
        end
        
    end
    
else
    a = 0.1;
    n = mesh_points;
    Dx = 1 / n;
    
    %% CONSTRUCT MATRIX MODEL AND COMPUTE DATA _WITH_ INVERSE CRIME
    % Construct normalized discrete point spread function
    nPSF = ceil(a / Dx); % number points (v+1) Dx > a
    %-- NUMBER OF POINTS THAT ONE OF PSF LYING IN f(x)
    xPSF = [-nPSF:nPSF] * Dx; % Coordinate of points
    %-- RELATIVE COORDITNATE OF ONE PSF
    PSF = zeros(size(xPSF));
    %-- PSF
    ind = abs(xPSF) < a; % (v+1) Dx > a => v
    %-- INDEX
    PSF(ind) = DC_PSF(xPSF(ind), a); % psf = (x + a)^2*(x - a)^2;
    % Ca = ( \int_{-a}^{a} [(x + a)^2*(x - a)^2] dx )^{-1)
    Ca = 1 / (Dx * trapz(PSF));
    PSF = Ca * PSF;
    %-- PSF AT MUTUAL POINTS
    
    % Construct convolution matrix
    A = Dx * DC_convmtx(PSF, n);
    A = sparse(A);
end

end

function [a] = kernel(t, s, beta)
a = (1 / sqrt(2 * pi * beta^2)) * exp(-(t - s)^2 / (2 * beta^2));
end

function mtx = DC_convmtx(PSF, n)

% Force PSF to be horizontal vector
PSF = PSF(:).';

% Determine the constant m (length of PSF should be 2*m+1)
m = (length(PSF) - 1) / 2;

% First: construct convolution matrix with the boundary condition assuming
% zero signal outside the boundaries
tmp = conv_mtx(PSF, n);
mtx = tmp(:, (m + 1):(end - m));

% Second: correct for the periodic boundary condition
for iii = 1:m
    mtx(1:m, (end - m + 1):end) = tmp(1:m, 1:m);
    mtx((end - m + 1):end, 1:m) = tmp((end - m + 1):end, (end - m + 1):end);
end

end

function psf = DC_PSF(x, a)

% Check that parameter a is in acceptable range
if (a <= 0) | (a >= 1/2)
    error('Parameter a should satisfy 0<a<1/2')
end

% Evaluate the polynomial
psf = (x + a).^2 .* (x - a).^2;

end

function X = conv_mtx(x, nh)
x1 = x(:);
p = length(x1);
X = zeros(p + nh - 1, nh);

for k = 1:nh
    X(k:p + (k - 1), k) = x1;
end

[pp, qq] = size(x);

if pp < qq
    X = X.';
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