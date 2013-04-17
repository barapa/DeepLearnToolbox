% if opt_nn is specified, the activations of the input will remain 
% in the returned nn.
function [labels opt_nn] = nnpredict(nn, x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    [~, i] = max(nn.a{end},[],2);
    labels = i;
    opt_nn = nn;
end
