function entropy = q4_entropy(Y)
% Compute the entropy for the label set Y

% INPUT
%  Y     : [m x 1] vector, the labels of the subset of examples for which entropy must
%          be computed

% OUTPUT
%  entropy : entropy of label set Y

P_c = zeros(1,2);
P_c(1,1) = size(find(Y == 0),1) / size(Y,1);
P_c(1,2) = size(find(Y == 1),1) / size(Y,1);


if (P_c(1,1) == 0 || P_c(1,2) == 0)
    entropy = 0;
else
    entropy = - P_c(1,1) *log(P_c(1,1)) - P_c(1,2) * log(P_c(1,2));
end
%    entropy = - log( P_c(1,1) ^ P_c(1,1)) -log(P_c(1,2) ^ P_c(1,2));

end
