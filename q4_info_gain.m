function info_gain = q4_info_gain(X, Y, idx)
% Compute the information gain on (X, Y) using feature idx as a split

% INPUT
%  X      : [m x n] matrix, where each row is an n-dimensional input example
%  Y      : [m x 1] vector, where the i-th element is the label for the i-th example
%  idx    : [1 x 1] scalar, feature index to be used as the split test

% OUTPUT
%  info_gain: [1 x 1] scalar, information gain on (X, Y) using the split
%                     test of feature idx.
%                     *** NOTE *** omit the first term of the
%                     information gain because it is a constant for a
%                     given set (X,Y)

%fprintf( 'X(, idx) == 1 count: %d\n', size(find(X(:, idx) == 1), 1) );
%fprintf( 'X count: %d\n', size(X, 1));
P_s = size(find(X(:, idx) == 1), 1) / size(X,1);
i_s = q4_entropy(Y(find(X(:, idx) == 1)));
i_sn = q4_entropy(Y(find(X(:, idx) == 0)));
info_gain = - P_s * i_s - (1-P_s) * i_sn;

end
