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

            v_raw = cell(cdk + 1, 1);
            v_sample = cell(cdk + 1, 1);

            h_raw = cell(cdk + 1, 1);
            h_sample = cell(cdk + 1, 1);

            v_raw{1} = batch;
            v_sample{1} = batch;

            h_raw{1} = repmat(rbm.c', batchsize, 1) + v_raw{1} * rbm.W';
            h_sample{1} = sigmrnd(h_raw{1});


            for k = 2 : cdk + 1
              v_raw{k} = repmat(rbm.b', batchsize, 1) + h_sample{k-1} * rbm.W;
              if rbm.gaussian_visible_units
                v_sample{k} = mvnrnd(v_raw{k}, eye(size(v_raw{k}, 2)));
              else
                v_sample{k} = sigmrnd(v_raw{k});
              end
              h_raw{k} = repmat(rbm.c', batchsize, 1) + v_sample{k} * rbm.W';
              h_sample{k} = sigmrnd(h_raw{k});
            end % for k = 1 : cdk + 1

            % Currently, we use p(v | h) directly instead of its sample.
            % switch h_sample to h_raw to use raw h for updates, which Hinton
            % says can speed up learning. p(h | v)
            phase_pos = h_sample{1}' * v_raw{1}; 
            phase_neg = h_sample{cdk + 1}' * v_raw{cdk + 1};

            rbm.vW = ...
              rbm.momentum * rbm.vW + rbm.alpha * (phase_pos - phase_neg) / batchsize;
            rbm.vb = ...
              rbm.momentum * rbm.vb + rbm.alpha * sum(v_raw{1} - v_raw{cdk + 1})' / batchsize;
            rbm.vc = ...
              rbm.momentum * rbm.vc + rbm.alpha * sum(h_sample{1} - h_sample{cdk + 1})' / batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v_raw{1} - v_raw{cdk + 1}) .^ 2)) / batchsize ;
            numbatches = numbatches + 1 ;
        end

        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);

    end
end
