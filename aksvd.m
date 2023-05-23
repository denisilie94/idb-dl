function [D, X, errs, coh, train_time] = aksvd(Y, D, n_nonzero_coefs, max_iter)
    % start waitbar
    train_time = 0;
    % wb = waitbar(0, 'Training AKSVD...');

    errs = zeros(1, max_iter);
    n_signals = size(Y, 2);
    [n_features, n_components] = size(D);

    for i_iter = 1:max_iter
        tmp_time = tic;

        % X coding method
        X = omp(Y, D, n_nonzero_coefs);

        % optimize dictionary D
        E = Y - D*X;
        for j = 1:n_components
            [~, data_indices, x] = find(X(j,:));

            if (isempty(data_indices))
                d = randn(n_features, 1);
                D(:, j) = d / norm(d);
            else
                F = E(:, data_indices) + D(:, j)*x;
                d = F*x';
                D(:, j) = d / norm(d);
                X(j, data_indices) = F'*D(:, j);
                E(:, data_indices) = F - D(:, j)*X(j, data_indices);
            end
        end

        errs(i_iter) = norm(Y - D*X, 'fro') / sqrt(n_features*n_signals);
        train_time = train_time + toc(tmp_time);

        % update waitbar
        % waitbar(i_iter/max_iter, wb, sprintf('Training AKSVD - Remaining time: %d [sec]',...
        %        round(train_time/i_iter*(max_iter - i_iter))));
    end

    % close waitbar
    % close(wb);

    coh = triu(abs(D'*D), 1);
    coh = sort(coh(:));
    coh = coh(n_components * (n_components + 1) / 2 + 1 : end);
end
