function [label, posterior] = q4_dt_predict(tree, X)
% Predicts the labels and computes the class posterior probabilities of the test
% examples in X

% INPUT
%  tree     : [L x 3] matrix, the learned tree. L is the number of nodes in the tree. 
%  X        : [m x n] matrix, where each row is an n-dimensional input example

% OUTPUT
%  label    : [m x 1] vector, the predicted labels of the test examples 
%  posterior: [m x 1] vector, the class posterior probabilities of the test examples

label = zeros(size(X,1),1);
posterior = zeros(size(label));

for i = 1: size(X,1)
    cur = 1;
    
    while (tree(cur,1) ~= 0)
        idx = tree(cur,1);
        if (X(i,idx) == 1)
            cur = tree(cur,2);
        else
            cur = tree(cur,3);
        end
    end
    
    label(i,1) = tree(cur, 2);
    posterior(i,1) = tree(cur, 3);
end

end
