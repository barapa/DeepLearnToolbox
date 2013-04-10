function X = sigmrnd(P)
%   P: vector
%   This takes a vector, puts it through an element sigmoid function and
%   samples a random binary variable from the probability binomial
%   distribution defined by the sigmoid output
%     X = double(1./(1+exp(-P)))+1 > rand(size(P));
    X = double(1./(1+exp(-P)) > rand(size(P)));
end