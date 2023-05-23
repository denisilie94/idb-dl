clc;
clear;
close all;

% DL parameters
n_samples = 1000;       % number of signals
n_features = 64;        % size of signals
n_components = 100;     % number of atoms
n_nonzero_coefs = 5;    % sparsity level

M = 1.2;                % barrier margin
gamma = 0.5;            % trade-off factors
lambda = 25;

noise_var = 0.01;
x_threshold = 0.2;
max_iter = 100;
n_rounds = 10;

errs_aksvd = zeros(n_rounds, max_iter);
coh_aksvd = zeros(n_rounds, (n_components - 1)*n_components/2);

errs_idb = zeros(n_rounds, max_iter);
coh_idb = zeros(n_rounds, (n_components - 1)*n_components/2);

for i_round = 1:n_rounds
    rng(i_round);

    % Prepare the true dictionary
    Dt = randn(n_features, n_components);
    Dt = normc(Dt);
    
    % Prepare the set of samples
    Y = zeros(n_features, n_samples);
    for i = 1:n_samples
        support = randperm(n_components);
        support = support(1:n_nonzero_coefs);

        x = randn(n_nonzero_coefs, 1);
        x = x + sign(x)*x_threshold;

        Y(:,i) = Dt(:, support)*x + noise_var*randn(n_features, 1);
    end

    % Init D0
    D0 = normcol_equal(randn(n_features, n_components));

    % AK-SVD method
    [~, ~, errs, coh, ~] = aksvd(Y, D0, n_nonzero_coefs, max_iter);
    coh_aksvd(i_round, :) = coh;
    errs_aksvd(i_round, :) = errs;
    
    % IDB method
    [~, ~, errs, coh, ~] = idb_dl(Y, D0, n_nonzero_coefs, M, gamma, lambda, max_iter);
    errs_idb(i_round, :) = errs;
    coh_idb(i_round, :) = coh;
end

csize = 14;
figure();
stdshade(errs_aksvd, 0.2, 'red');
hold on; grid on;
stdshade(errs_idb, 0.2, 'blue');
xlabel('iteration', 'interpreter', 'latex', 'FontSize', csize)
ylabel('error', 'interpreter', 'latex', 'FontSize', csize)
h = legend('', 'AK-SVD', '', 'IDB-DL');
set(h, 'interpreter', 'latex', 'FontSize', csize);
pbaspect([1, 0.5, 1]);
                
figure();
plot(mean(coh_aksvd), 'red')
hold on; grid on;
plot(mean(coh_idb), 'blue')
xlabel('\# product', 'interpreter', 'latex', 'FontSize', csize)
ylabel('atom scalar products', 'interpreter', 'latex', 'FontSize', csize)
h = legend('AK-SVD', 'IDB-DL');
set(h, 'interpreter', 'latex', 'FontSize', csize);
pbaspect([1, 0.5, 1]);
