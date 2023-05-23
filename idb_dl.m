function [D, X, errs, coh, train_time] = aksvd_coh_dist(Y, D, n_nonzero_coefs, M, gamma, lambda, max_iter)
    % start waitbar
    train_time = 0;
    % wb = waitbar(0, 'Training AKSVD coh dist...');
    
    errs = zeros(1, max_iter);
    n_signals = size(Y, 2);
    [n_features, n_components] = size(D);
    desired_coh = 1 - M/2;

    for i_iter = 1:max_iter
        tmp_time = tic;
        
        % sparse coding
        X = omp(Y, D, n_nonzero_coefs);

        % dictionary update
        E = Y - D*X;
        for j = 1:n_components   
            [~, data_indices, x] = find(X(j,:));

            if (isempty(data_indices))
                d = randn(n_features, 1);
                D(:, j) = d / norm(d);
            else
                % scalar products and weights
                d = D(:,j);
                v = D'*d;
                v(j) = 0;
                w = max(abs(v)/desired_coh, 1);

                % what atoms are too close?
                i_minus = find(v>desired_coh);
                i_plus = find(v<-desired_coh);

                % gradients
                g1 = D*(w.*v);
                g2 = zeros(n_features, 1);
                if ~isempty(i_minus)
                    g2 = g2 + sum(D(:,i_minus) - repmat(d,1,length(i_minus)), 2);
                end
                if ~isempty(i_plus)
                    g2 = g2 - sum(D(:,i_plus) + repmat(d,1,length(i_plus)), 2);
                end

                F = E(:, data_indices) + d * x;
                d = F*x' - gamma*(g1 + lambda*g2);

                D(:, j) = d / norm(d);
                X(j, data_indices) = F'*D(:, j);
                E(:, data_indices) = F - D(:, j)*X(j, data_indices);
            end
        end

        errs(i_iter) = norm(Y - D*X, 'fro') / sqrt(n_features*n_signals);
        train_time = train_time + toc(tmp_time);

        % update waitbar
        % waitbar(i_iter/max_iter, wb, sprintf('Training AKSVD coh dist - Remaining time: %d [sec]',...
        %        round(train_time/i_iter*(max_iter - i_iter))));
    end
    
    % close waitbar
    % close(wb);

    coh = triu(abs(D'*D), 1);
    coh = sort(coh(:));
    coh = coh(n_components * (n_components + 1) / 2 + 1 : end);
end
