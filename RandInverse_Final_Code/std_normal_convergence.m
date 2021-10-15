% A brief study of the convergence of estimating the covariance of a standard normal distribution
% using Monte Carlo approximations. 

samples.n_samples = [100, 1000, 10000, 50000];
samples.dim = 1000;
samples.n_samples_percentage = samples.n_samples / samples.dim * 100;
for i=1:numel(samples.n_samples)
		  samples.data{i} = normrnd(0, 1, samples.dim, samples.n_samples(i));
          samples.covariance{i} = samples.data{i} * samples.data{i}' / samples.n_samples(i);
          samples.singular_vals{i} = svd(samples.covariance{i});
end

color = zeros(4,3);
color(1,:) = [0 0.4470 0.7410];
color(2,:) = [0.6350 0.0780 0.1840];
color(3,:) = [0.4660 0.6740 0.1880];
color(4,:) = [204.0, 102.0, 0.0] / 255.0;

marker_style{1} = '-o';
marker_style{2} = '-x';
marker_style{3} = '-d';
marker_style{4} = '-^';

marker_increment = 100;
marker_indices = 1:marker_increment:samples.dim;

plot([1:samples.dim], ones(samples.dim, 1), '--k', 'Linewidth', 1,'DisplayName','true  $\mathcal{C}^{-1}$'); hold on
for i=1:numel(samples.n_samples)
        h = plot([1:samples.dim], samples.singular_vals{i}, marker_style{i}, 'color',color(i,:), 'Linewidth', 1, ...
'DisplayName',['N = ' num2str(samples.n_samples(i))], 'MarkerSize', 10, 'MarkerIndices', marker_indices);
        legend('Location','northeast', 'Interpreter','latex')
        set(h, 'MarkerFaceColor', get(h, 'Color'));
end
title('Spectrum of sample inverse covariance \lambda \lambda^T')
ylim([-0.25, 5.0])
set(findall(gcf,'-property','FontSize'),'FontSize',12,'FontName', 'Times New Roman')
saveas(gcf,'covariance_convergence','epsc')
