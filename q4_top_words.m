function word_idx = q4_top_words(phi_y0, phi_y1, phi_prior, k)
% For each class, finds the words that are most indicative of a message
% belonging to that class

% INPUT
%  phi_y0    : [n x 1] vector, class conditional probabilities for y=0,
%              where phi_y0(j) = p(x_j = 1 | y = 0)
%  phi_y1    : [n x 1] vector, class conditional probabilities for y=1, 
%              where phi_y0(j) = p(x_j = 1 | y = 1)
%  phi_prior : [1 x 1] scalar, prior probability of y being 1, i.e., phi_prior = p(y = 1)
%  k         : [1 x 1] scalar, the number of words to output

% OUTPUT
%  word_idx  : [2 x k] matrix, the first row contains the indices of the k most indicative 
%              words for class y=0, the the second row the ones for y=1

word_idx = zeros(2,k);
y0 = zeros(size(phi_y0));
y1 = zeros(size(phi_y1));
for i = 1: size(phi_y0,1)
    y0(i,1) = phi_y0(i,1) * (1-phi_prior) / (phi_y0(i,1)*(1-phi_prior) + phi_y1(i,1)*phi_prior);
    y1(i,1) = 1-y0(i,1);
end
[val, loc] = sort(y0, 'descend');
word_idx(1, :) = loc(1:k,1);
[val, loc] = sort(y1, 'descend');
word_idx(2, :) = loc(1:k,1);
end
