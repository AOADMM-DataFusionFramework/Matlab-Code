function p = t_smoothness_prox(factor_matrices,rho,smoothness_l)

rhs = cell(length(factor_matrices),1);

% disp(class(factor_matrices));
% disp(class(rho));

for i = 1:length(factor_matrices)
    rhs{i} = rho(i) * factor_matrices{i};
end

% Construct the (extended) matrix A to perform thomas algorithm on
% A = [ a  c  0  0  0  0     |  d
%       c  b  c  0  0  0     |  d
%       0  c  b  c  0  0     |  d
%       0  0  c  b  c  0     |  d
%       0  0  0  c  b  c     |  d
%       0  0  0  0  c  a     |  d ]
%
% where a = 2 * smoothness_l + rho(i), b = 4 * smoothness_l + rho(i), 
% c = -2 * smoothness_l and d = rhs(i) (i.e. rho(i) * (G.B(i) + G.mu_B_Z(i)))

A = zeros(length(rho),length(rho));

for i=1:length(rho)
    for j=1:length(rho)
        if i == j
            A(i,j) = 4 * smoothness_l + rho(i);
        elseif i == j - 1 || i == j + 1
            A(i,j) = -2 * smoothness_l;
        end
    end
end

A(1,1) = A(1,1) - 2 * smoothness_l;
A(end,end) = A(end,end) - 2 * smoothness_l;

% Perform GE

for i=2:length(rho)
    m = A(i,i-1) / A(i-1,i-1);
    A(i,i) = A(i,i) - m * A(i-1,i);
    rhs{i} = rhs{i} - m * rhs{i-1};
end

% Back-substitution
new_ZBks = cell(1, length(factor_matrices));
new_ZBks{end} = rhs{end} / A(end,end);
q = new_ZBks{end};

for k=length(rho)-1:-1:1
    q = (rhs{k} - A(k,k+1) * q) / A(k,k);
    new_ZBks{k} = q;
end

p = new_ZBks;

end

