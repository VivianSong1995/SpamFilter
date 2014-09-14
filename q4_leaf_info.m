function [label, posterior] = q4_leaf_info(Y)
% Compute the majority class label and the posterior at the leaf node given
% the labels Y of the training examples at this node

% INPUT
%  Y        : [m x 1] vector, labels of the training examples at this node

% OUTPUT
%  label    : [1 x 1] scalar, majority label at this leaf node
%  posterior: [1 x 1] scalar, posterior probability of Y being 1 at this
%             leaf node. This is given by the fraction of examples having label 1 in Y


P_c = zeros(2,1);
P_c(1,1) = size(find(Y == 0),1) / size(Y,1);
P_c(2,1) = size(find(Y == 1),1) / size(Y,1);
if P_c(1,1) > P_c(2,1)
    label = 0;
else
    label = 1;
end
posterior = P_c(2,1);

end
