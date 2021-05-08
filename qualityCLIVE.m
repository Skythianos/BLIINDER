clear all
close all
 
load CLIVE.mat    % This mat file contains information about the database (CLIVE)
 
Directory = '/home/domonkos/Desktop/QualityAssessment/Databases/CLIVE/ChallengeDB_release/ChallengeDB_release/Images';  % path to CLIVE database 
numberOfImages = size(AllMOS_release,2);   % number of images in CLIVE database
numberOfTrainImages = round( 0.8*numberOfImages );   % appx. 80% of images is used for training
numberOfSplits = 100;
 
j=1;
Constants.net = vgg16;    % alexnet, vgg16, vgg19
for i=2:size(Constants.net.Layers,1)-1
     tmp=Constants.net.Layers(i);
     if(contains(tmp.Name, 'drop'))
         
     else
         Constants.Layers{j}=tmp.Name;
         if(isprop(tmp,'Bias'))
             Constants.Lengths{j}=size(tmp.Bias,3);
         else
             Constants.Lengths{j}=Constants.Lengths{j-1};
         end
         j=j+1;
     end 
end
 
Features = cell(1, length(Constants.Layers));

disp('Feature extraction');
for ii=1:length(Constants.Layers)
     disp(ii);
     tmpFeatures = zeros(numberOfImages, 2*Constants.Lengths{ii});
     for i=1:numberOfImages
         if(mod(i,100)==0)
             disp(i);
         end
         imgDist          = imread( strcat(Directory, filesep, AllImages_release{i}) );
         tmpFeatures(i,:) = getDeepFeatures(imgDist, Constants, ii); 
     end 
     Features{ii} = tmpFeatures;
end

disp('Training and testing');
numberOfSplits=100;
PLCC =zeros(1,numberOfSplits);
SROCC=zeros(1,numberOfSplits);
for i=1:numberOfSplits
    p = randperm(numberOfImages);
    Target = AllMOS_release(p);
    TrainLabel = Target(1:round(numberOfImages*0.8));
    TestLabel = Target(round(numberOfImages*0.8)+1:end);
    SubPred = cell(1, length(Constants.Layers));
    for ii=1:length(Constants.Layers)   
        tmpFeatures = Features{ii};
        Data = tmpFeatures(p,:);   
    
        trainFeatures = Data(1:round(numberOfImages*0.8),:);
        testFeatures  = Data(round(numberOfImages*0.8)+1:end,:);
    
        Mdl = fitrsvm(trainFeatures, TrainLabel, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
        SubPred{ii} = predict(Mdl, testFeatures);
    end
    TMP = cell2mat(SubPred);
    TMP = sum(TMP,2);
    Pred = TMP ./ length(Constants.Layers);
    beta(1) = max(TestLabel); 
    beta(2) = min(TestLabel); 
    beta(3) = mean(TestLabel);
    beta(4) = 0.5;
    beta(5) = 0.1;
    [bayta,ehat,J] = nlinfit(Pred',TestLabel,@logistic,beta);
    [pred_test_mos_align, ~] = nlpredci(@logistic,Pred,bayta,ehat,J);
    
    PLCC(i) = corr(pred_test_mos_align,TestLabel');
    SROCC(i)= corr(Pred,TestLabel','Type','Spearman');
end
