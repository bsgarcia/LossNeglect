function [xopt, fval, exitflag] = optimize(funcname, x0, A, b, ...
    Aeq, beq, lb, ub, nonlcon, opt_struct, gradients, optional_args)

    % Written by Andrew Ning.  Feb 2016.
    % FLOW Lab, Brigham Young University.

    % set options
    options = optimoptions('fmincon');
    names = fieldnames(opt_struct);
    for i = 1:length(names)
        options = optimoptions(options, names{i}, opt_struct.(names{i}));
    end

%
    % check if gradients provided
%    if gradients
%        options = optimoptions(options, 'GradObj', 'on', 'GradConstr', 'on');
%    end

    % run fmincon
    [xopt, fval, exitflag] = fmincon(@obj, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    % ---------- Objective Function ------------------
    function [xopt] = obj(x0)
        eval(['xopt = py.', funcname, '(x0, optional_args);'])
    end
end


