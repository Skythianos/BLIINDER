function [output] = getDeepFeatures(img ,Constants, ind)
    actLayer = Constants.Layers{ind};
    actValues= activations(Constants.net, img, actLayer);
    output = zeros(1, 2*Constants.Lengths{ind});
    for i=1:2*Constants.Lengths{ind}
        if(i<=Constants.Lengths{ind})
            tmp = actValues(:,:,i);
            output(i) = min(tmp(:));
        else
            tmp = actValues(:,:,i-Constants.Lengths{ind});
            output(i) = max(tmp(:)); 
        end
    end
end

