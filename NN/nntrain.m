function [nn, L, opts]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
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

if isfield(opts, 'loss')
  loss.train.e = opts.loss.train.e;
  loss.train.e_frac = opts.loss.train.e_frac;
  loss.val.e = opts.loss.val.e;
  loss.val.e_frac = opts.loss.val.e_frac;
else
  loss.train.e               = [];
  loss.train.e_frac          = [];
  loss.val.e                 = [];
  loss.val.e_frac            = [];
end

if ~isfield(opts, 'i')
  opts.i = 0;
end

opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts, 'figure')
    fhandle = opts.figure;
elseif isfield(opts, 'plot') && opts.plot == 1
    fhandle = figure();
    opts.figure = fhandle;
end

if ~isfield(opts, 'batch')
  opts.batch = 0;
else
  opts.batch = opts.batch + 1;
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = ceil(m / batchsize);

L = [ ] ;


for i = 1 : numepochs
    tic;

    kk = randperm(m);
    epoch_L = [ ] ;
    for l = 1 : opts.batchsize : m
        if l + opts.batchsize < length(kk)
          batch_x = train_x(kk(l : l + opts.batchsize - 1), :) ;
          batch_y = train_y(kk(l : l + opts.batchsize - 1), :) ;
        else
          batch_x = train_x(kk(l : end), :) ;
          batch_y = train_y(kk(l : end), :) ;
        end

        % Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end

        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);

        L = [ L, nn.L ] ;
        epoch_L = [ epoch_L, nn.L ] ;
    end

    t = toc;

    if ishandle(fhandle)
        if opts.validation == 1
            loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        else
            loss = nneval(nn, loss, train_x, train_y);
        end
        opts.loss = loss;
        opts.i = opts.i + 1;
        nnupdatefigures(nn, fhandle, loss, opts, opts.i);
    end

    % TODO(sam): fix error message
    disp(['Training NN: epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' ...
        num2str(t) ' seconds' '. Mean squared error on training set is ' ...
        num2str(mean(epoch_L))]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end
end

