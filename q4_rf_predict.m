function [label, posterior] = q4_rf_predict(treeset, X)
% Predicts the labels and computes the class posterior probabilities of the test
% examples in X

% INPUT
%  tree     : [L x 3] matrix, the learned tree. L is the number of nodes in the tree. 
%  X        : [m x n] matrix, where each row is an n-dimensional input example

% OUTPUT
%  label    : [m x 1] vector, the predicted labels of the test examples *based on the class majority scheme*
%  posterior: [m x 1] vector, the *average* class posterior probabilities of the test examples

labels = zeros(size(X,1), size(treeset,1)); %m x k
posteriors = zeros(size(X,1), size(treeset,1));
for i = 1: size(treeset,1)
	[labels(:,i), posteriors(:,i)] = q4_dt_predict(treeset{i}, X);
end

label = zeros(size(X,1),1);
posterior = zeros(size(X,1),1);
for i = 1: size(X,1)
    labels_i = labels(i, :);
    posteriors_i = posteriors(i, :);
    pred = unique(labels_i);
    count = zeros(size(pred));
    for j = 1:size(pred,2)
        count(1,j) = size(find(labels_i == pred(1,j)),2);
    end

    [m, loc] = max(count);
    label(i,1) = pred(1,loc); 
    posterior(i,1) = mean(posteriors_i);
end

end
