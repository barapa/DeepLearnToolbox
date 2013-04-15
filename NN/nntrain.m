function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nntrain(nn, x, y, opts, val_x, val_y) trains the neural network 
% nn with input x and output y for opts.numepochs epochs, with minibatches
% of size opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.
%
% train_x : N x D matrix
% train_y : N x L matrix, where L is the number of labels for one-hot
%           encoding
% val_x :   (optional) N x D matrix to be used for validation, but not
%           training
% val_y :   (optional) N x L matrix to be used for validation, but not
%           training

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

% this check is no longer necessary
%assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_start_ind = (l - 1) * batchsize + 1;
        batch_end_ind = min(l * batchsize, numel(kk));
        batch_x = train_x(kk(batch_start_ind : batch_end_ind), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk(batch_start_ind : batch_end_ind), :);
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    t = toc;
    
    if ishandle(fhandle)
        if opts.validation == 1
            loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        else
            loss = nneval(nn, loss, train_x, train_y);
        end
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end
end

