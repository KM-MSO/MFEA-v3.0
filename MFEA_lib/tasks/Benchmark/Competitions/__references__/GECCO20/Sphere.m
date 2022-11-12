function obj = Sphere(x,M,opt)
%Sphere function
%   - var: design variable vector
%   - M: rotation matrix
%   - opt: shift vector
% x = decode(x)
    var = x;
    dim = length(var);
    var = (M*(var-opt)')';


% var = myDecode(x)
    sum = 0;
    for i = 1: dim
        sum = sum + var(i)*var(i);
    end
    obj = sum;    
end

% myEncode(var)
% var = invM*var + opt
% var = scale(0 1) (var)