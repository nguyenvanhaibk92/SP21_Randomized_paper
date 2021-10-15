% Input the size of the parameter and a scaling factor
% The function will return a prior covariance matrix implementing
% centered difference for ||\grad u||^2

function prior = smoothness_prior(parameter_size, scaling_factor)
    prior = diag(2*ones(1, parameter_size));
    prior = prior + diag(-ones(1, parameter_size-1), 1);
    prior = prior + diag(-ones(1, parameter_size-1), -1);
    prior = scaling_factor * prior;
end
