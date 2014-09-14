function feat_selected = q4_rf_split(X, Y, feat_idx, F)
% Find the split that maximizes the information gain for the subset 
% (X, Y) of the training set from a random subset of F features

% INPUT

%  X        : [m x n] matrix, where each row is an n-dimensional input example
%  Y        : [m x 1] vector, where the i-th element is the label for the i-th example
%  feat_idx : [1 x L] vector, indices of features to be potentially considered
%  F        : [1 x 1] scalar, size of the random subset of features to be considered

% OUTPUT
%  feat_selected : [1 x 1] scalar, the feature that maximizes the information gain for (X, Y) 
%                  (this should be one of the numbers stored in feat_idx and an integer between 1 and n). 
%                  Note this value must be set to 0 if there is no feasible split


% INSERT YOUR CODE HERE:
% compute the feasible feature indices
% store them in feasible_idx

feasible_idx = [];
count = 1;
for i = 1: size(feat_idx,2)
    sz1 = size(find(X(:, feat_idx(1,i)) == 1),1);
    sz2 = size(find(X(:, feat_idx(1,i)) == 0),1);
    if (sz1 ~= 0 && sz2 ~= 0)
        feasible_idx(1,count) = feat_idx(1,i);
        count = count + 1;
    end
end
% feasible_idx = feasible_idx(1, 1:count-1);
if isempty(feasible_idx)
    feat_selected = 0;
else
%feasible_idx
% USE THIS AS RANDOM SELECTION OF FEATURE SUBSET
% LEAVE UNCHANGED
% -----------------------------------------------
    indperm = randperm(length(feasible_idx));
    if length(feasible_idx)>F
        feasible_idx = feasible_idx(indperm(1:F));
    end
% -----------------------------------------------

% INSERT YOUR CODE HERE:
% choose within this subset the one with the best gain
%feasible_idx
    info_gain = zeros(size(feasible_idx));
    for i = 1: size(feasible_idx,2)
        info_gain(1,i) = q4_info_gain(X, Y, feasible_idx(1,i));
    end
    [val, loc] = max(info_gain);
    feat_selected = feasible_idx(1, loc);
end

end
