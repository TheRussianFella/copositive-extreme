function [s,x] = is_copositive(A)
% function [s,x] = is_copositive(A)
%
% checks the real symmetric matrix A for copositivity
% s is a boolean indicating copositivity
% if s is false, then x is a nonzero vector in R_+ such that x'Ax < 0
n = size(A,1);
x = zeros(n,1);
if n == 1,
    if A >= 0,
        s = true;
        return;
    end,
    s = false;
    x = 1;
    return;
end,
[U,D] = eig(A);
d = diag(D);
[mi,in] = min(d);
tol = max(d) * max(size(A)) * eps;
%disp(tol);
%disp(mi);
if mi >= -tol,
    s = true;
    return;
end,
u = U(:,in);
if min(u) < tol,
    u = -u;
end,
if min(u) > tol,
    s = false;
    x = u;
    return;
end,
for k = 1:n,
    in = 1:n;
    in(k) = [];
    B = A(in,in);
    [s2,x2] = is_copositive(B);
    if s2 == false,
        s = false;
        x = [x2(1:k-1); 0; x2(k:end)];
        return;
    end,
end,
s = true;
