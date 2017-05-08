files = dir('*.mat');
for class = 1:7
    for sample = 1:20
        f = sprintf('Class%i_Sample%i.mat', class, sample);
        load(f);
        N = size(x, 1);
        clf;
        scatter(x(:, 1), x(:, 2), 20, 1:size(x, 1), 'fill');
        axis equal;
        title(f);
        [a, b] = ginput(2);
        y = [a b];
        D = pdist2(y, x);
        [~, idx] = min(D, [], 2);
        
        %Figure out if going clockwise or counterclockwise
        i1 = [idx(1), idx(1) + N];
        i2 = [idx(2), idx(2) + N];
        D = abs(bsxfun(@minus, i1(:), i2));
        [~, idx] = min(D(:));
        [idx1, idx2] = ind2sub(size(D), idx);
        idx1 = i1(idx1);
        idx2 = i2(idx2);
        inorder = 1;
        if idx2 - idx1 < 0
            inorder = 0;
        end
            
        
        %Now reorder indices accordingly
        idx = mod(idx1, N);
        if idx == 0
            idx = N;
        end
        if inorder
            x1 = x(1:idx, :);
            x2 = x(idx+1:end, :);
            x = [x2; x1];
        else
            x1 = flipud(x(1:idx, :));
            x2 = flipud(x(idx+1:end, :));
            x = [x1; x2];
        end
        scatter(x(:, 1), x(:, 2), 20, 1:size(x, 1), 'fill');
        axis equal;
        pause;
        save(f, 'x');
    end
end

%[1, 10; 