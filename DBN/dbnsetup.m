function dbn = dbnsetup(dbn, x, opts)
    rand_weight_sigma = 0.001;
    
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = mvnrnd(zeros(dbn.sizes(u + 1), dbn.sizes(u)),...
            rand_weight_sigma * eye(dbn.sizes(u)));
        % vW holds the delta for each weight update, for use with momentum
        % calculations
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = mvnrnd(zeros(dbn.sizes(u), 1), rand_weight_sigma);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = mvnrnd(zeros(dbn.sizes(u + 1), 1),...
            rand_weight_sigma);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
        
        % set the lowest rbm to gaussian if specified
        if u == 1 && dbn.gaussian_visible_units
            dbn.rbm{u}.gaussian_visible_units = 1;
        else
            dbn.rbm{u}.gaussian_visible_units = 0;
        end
    end

end
