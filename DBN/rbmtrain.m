function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    m = size(x, 1);
    assert(isfield(opts, 'cdk'), 'must define cdk in training params!');
    assert(opts.cdk >= 1, 'cdk must be integer >= 1!');
    cdk = opts.cdk;


    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        numbatches = 0;
        for l = 1 : opts.batchsize : m
            if l + opts.batchsize < length(kk)
              batch = x(kk(l : l + opts.batchsize - 1), :) ;
            else
              batch = x(kk(l : end), :) ;
            end

            batchsize = size(batch, 1);

            v_1 = batch;
            v_k = batch;
            v_k_raw = batch;
            h_1 = sigmrnd(repmat(rbm.c', batchsize, 1) + v_1 * rbm.W');
            h_k = h_1;

            c_1 = h_1' * v_1;

            for k = 1 : cdk
                h_k = sigmrnd(repmat(rbm.c', batchsize, 1) + v_k * rbm.W');
                v_k_raw = repmat(rbm.b', batchsize, 1) + h_k * rbm.W;

                if rbm.gaussian_visible_units
                  v_k = mvnrnd(v_k_raw, eye(size(v_k, 2))) ;
                else
                  v_k = sigmrnd(v_k_raw);
                end % if rbm.gaussian_visible_units

            end % for k = 1 : cdk

            % Currently, we use p(v | h) directly instead of its sample.
            c_k = h_k' * v_k_raw;

            rbm.vW = ...
                rbm.momentum * rbm.vW + rbm.alpha * (c_1 - c_k) / batchsize;
            rbm.vb = ...
                rbm.momentum * rbm.vb + rbm.alpha * sum(v_1 - v_k)' / batchsize;
            rbm.vc = ...
                rbm.momentum * rbm.vc + rbm.alpha * sum(h_1 - h_k)' / batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v_1 - v_k) .^ 2)) / batchsize ;
            numbatches = numbatches + 1 ;
        end

        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);

    end
end
