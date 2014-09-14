function feat_selected = q4_split(X, Y, feat_idx)
% Find the split that maximizes the information gain for the subset 
% (X, Y) of the training set

% INPUT
%  X      : [m x n] matrix, where each row is an n-dimensional input example
%  Y      : [m x 1] vector, where the i-th element is the label for the i-th example
%  feat_idx : [1 x L] vector, indices of features to be considered

% OUTPUT
%  feat_selected : [1 x 1] scalar, the feature that maximizes the information gain for (X, Y) 
%                  (this should be one of the numbers stored in feat_idx and an integer between 1 and n)

feasible_idx = zeros(size(feat_idx));
count = 1;
for i = 1: size(feat_idx,2)
    sz1 = size(find(X(:, feat_idx(1,i)) == 1),1);
    sz2 = size(find(X(:, feat_idx(1,i)) == 0),1);
    if (sz1 ~= 0 && sz2 ~= 0)
        feasible_idx(1,count) = feat_idx(1,i);
        count = count + 1;
    end
end
feasible_idx = feasible_idx(1, 1:count-1);
if (count == 1)
    feat_selected = 0;
else

    info_gain = zeros(size(feasible_idx));
    for i = 1: size(feasible_idx,2)
        info_gain(1,i) = q4_info_gain(X,Y,feasible_idx(1,i));
    end
    %info_gain
    [val, loc] = max(info_gain);
    feat_selected = feasible_idx(1,loc);
end

end
