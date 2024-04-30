
%% Main Function: Run This to produce assignment results
function[] = assignment3()
%% Set 1
    % load the dataset
    load Assignment_data.mat
    
    % template for dataset 1 results
    result_table = array2table(zeros(5,13),'VariableNames',{'Variable #', ...
    'R2 test PCR', 'R2 test PCR WOO', 'R2 test PLSR', 'R2 test PLSR WOO', 'R2 test PLSR Var Sel 1', 'R2 test PLSR Var Sel 2',...
    'RMSE test PCR', 'RMSE test PCR WOO', 'RMSE test PLSR','RMSE test PLSR WOO','RMSE test PLSR Var Sel 1','RMSE test PLSR Var Sel 2'});

    % assing response numbers
    result_table.("Variable #") = [1;2;3;4;5];


    %Assign training and test data
    xTrain = Set1.cal_X;
    yTrain = Set1.cal_Y;
    xTest = Set1.pred_X;
    yTest = Set1.pred_Y;
    
    
    %Are the predictors correlated?
    figure;
    h = heatmap(corrcoef(xTrain));
    h.GridVisible = 'off';
    %Some degree of correlation betwen adjacent and ends of the spectra
    
    %boxplots for responses
    f=figure('Renderer', 'painters', 'Position', [10 10 400 400]);
    boxplot(yTrain)
    xlabel('Response')
    ylabel('Intensity')
    fontname(f,"Times New Roman");
    saveas(f,'fig1.png')

    %find outliers in for each Y and remove outlier datapoints using the
    %quartile method
    outidxY = isoutlier(yTrain,"quartiles",1);
    xTrainWOO= xTrain(~max(outidxY,[],2),:);
    yTrainWOO= yTrain(~max(outidxY,[],2),:);
    %calculate PCs for X
    [xCoeff,xScore,xLatent,~,xExplained] = pca(xTrain,"Centered",true);

    %plot the variance explained by each PC
    f=figure('Renderer', 'painters', 'Position', [10 10 500 500]);
    bar(xExplained);
    sum(xExplained(1:5))
    xlabel('PC #')
    ylabel('% Variance explained')
    fontname(f,"Times New Roman");
    saveas(f,'fig1.5.png')
    rng("default")

    %plot the correlation plots for first 5 PCs and each 5 responses
    f=figure('Renderer', 'painters', 'Position', [10 10 1000 1000]);

    tiledlayout(5,5);
    for i=1:5
        for j=1:5
            nexttile;
            s = scatter(xScore(:,i), yTrain(:,j), 6, "filled", "MarkerFaceAlpha", 0.5);
            m = fitlm(xScore(:,i), yTrain(:,j));
            hold on
            plot(xScore(:,i), m.Fitted, "LineWidth", 2)
            text(min(xScore(:,i)).*0.8,max(yTrain(:,j))*0.8,strcat('R^2 = ',string(round(m.Rsquared.Adjusted,2))));
            if j==1
               ylabel(strcat('PC ', string(i)));
            end
            if i == 1
               title(append('Response ', string(j)),'FontWeight','normal');
            end
        end
    end
    fontname(f,"Times New Roman");
    saveas(f,'fig2.png')


    %for each of the responses tune and test regressions for PCR and PLSR
    for i=1:5
        %Tune and test PCR
        [~, ~, ~, result_table.("RMSE test PCR")(i), ~, result_table.("R2 test PCR")(i)] = pca_regression(xTrain,xTest,yTrain(:,i),yTest(:,i));
        
        %Tune and test PCR for dataset without outliers
        [~, ~, ~, result_table.("RMSE test PCR WOO")(i), ~, result_table.("R2 test PCR WOO")(i)] = pca_regression(xTrainWOO,xTest,yTrainWOO(:,i),yTest(:,i));

        %Tune and test PLSR
        [~, ~, ~, result_table.("RMSE test PLSR")(i), ~, result_table.("R2 test PLSR")(i)] = pls_regression(xTrain,xTest,yTrain(:,i),yTest(:,i));
       

        %Tune and test PLSR for dataset without outliers
        [~, ~, ~, result_table.("RMSE test PLSR WOO")(i), ~, result_table.("R2 test PLSR WOO")(i)] = pls_regression(xTrainWOO,xTest,yTrainWOO(:,i),yTest(:,i));
  
        %Tune and test PLSR for 10 variables selected based on FFS
        [~, ~, ~, result_table.("RMSE test PLSR Var Sel 1")(i), ~, result_table.("R2 test PLSR Var Sel 1")(i)] = pls_regression_fsrtest(xTrain,xTest,yTrain(:,i),yTest(:,i));
        
        %Tune and test PLSR for 10 variables selected based on MRMR
        [~, ~, ~, result_table.("RMSE test PLSR Var Sel 2")(i), ~, result_table.("R2 test PLSR Var Sel 2")(i)] = pls_regression_fsrmrmr(xTrain,xTest,yTrain(:,i),yTest(:,i));
        
    end
    
    %save the results table
    writetable(result_table,'res1.csv');


    %% Set 2
    res = zeros(2,4);

    %Assign training and test sets
    xTrain = Set2.cal_X;
    yTrain = Set2.cal_Y;
    xTest = Set2.pred_X;
    yTest = Set2.pred_Y;
    %Convert the responses to binary
    yTest = yTest - 1;
    yTrain = yTrain - 1;
    
    %Are the predictors correlated?
    figure;
    h = heatmap(corrcoef(xTrain));
    h.GridVisible = 'off';
    %Some degree of correlation betwen adjacent and ends of the spectra
    
    
    %find outliers in for each X and remove outlier datapoints
    outidxX = isoutlier(xTrain,"quartiles",1);
    xTrainWOO= xTrain(~max(outidxX,[],2),:);
    yTrainWOO= yTrain(~max(outidxX,[],2),:);
    size(yTrainWOO)
    %PCA of the xCal
    [xCoeff,xScore,xLatent,~,xExplained] = pca(xTrain,"Centered",true);
    sum(xExplained(1:5))
    %plot the variance explained by each PC
    f=figure('Renderer', 'painters', 'Position', [10 10 500 500]);
    bar(xExplained);
    xlabel('PC #')
    ylabel('% Variance explained')
    fontname(f,"Times New Roman");
    saveas(f,'fig3.png')
    rng("default")

    %plot the datapoints with their clases for first four pairwise PCs 
    f=figure('Renderer', 'painters', 'Position', [10 10 800 600]);
    tiledlayout(2,3);   
    nexttile;
    gscatter(xScore(:,1),xScore(:,2),yTrain);
    xlabel(strcat('PC1'));
    ylabel(strcat('PC2')) 
    nexttile;
    gscatter(xScore(:,1),xScore(:,3),yTrain);
    xlabel(strcat('PC1'));
    ylabel(strcat('PC3')) 
    nexttile;
    gscatter(xScore(:,1),xScore(:,4),yTrain);
    xlabel(strcat('PC1'));
    ylabel(strcat('PC4')) 
    nexttile;
    gscatter(xScore(:,2),xScore(:,3),yTrain);
    xlabel(strcat('PC2'));
    ylabel(strcat('PC3')) 
    nexttile;
    gscatter(xScore(:,2),xScore(:,4),yTrain);
    xlabel(strcat('PC2'));
    ylabel(strcat('PC4'));
    nexttile;
    gscatter(xScore(:,3),xScore(:,4),yTrain);
    xlabel(strcat('PC3'));
    ylabel(strcat('PC4'));
    fontname(f,"Times New Roman");
    saveas(f,'fig4.png')


    %train and test PCA classifier 
    [~,~,~,~,res(1,1)] = pca_classifier(xTrain,xTest,yTrain,yTest);
    %train and test PCA classifier for data without outlier
    [~,~,~,~,res(1,2)] = pca_classifier(xTrainWOO,xTest,yTrainWOO,yTest); 
    %train and test PCA classifier for 10 selected variables based on FFS
    [~,~,~,~,res(1,3)] = pca_classifier_fsrtest(xTrain,xTest,yTrain,yTest);
    %train and test PCA classifier for 10 selected variables based on MRMR
    [~,~,~,~,res(1,4)] = pca_classifier_fsrmrmr(xTrain,xTest,yTrain,yTest); 
    %train and test PLS classifier 
    [~,~,~,~,res(2,1)] = pls_classifier(xTrain,xTest,yTrain,yTest); 
    %train and test PLS classifier for data without outlier
    [~,~,~,~,res(2,2)] = pls_classifier(xTrainWOO,xTest,yTrainWOO,yTest); 
    %train and test PLS classifier for 10 selected variables based on FFS
    [~,~,~,~,res(2,3)] = pls_classifier_fsrtest(xTrain,xTest,yTrain,yTest); 
    %train and test PLS classifier for 10 selected variables based on MRMR
    [~,~,~,~,res(2,4)] = pls_classifier_fsrmrmr(xTrain,xTest,yTrain,yTest); 

    res = array2table(res);

    res.Properties.VariableNames = {'Tuned Model','Tuned Model Without Outliers', 'Tuned Model on 10 Variables Selected by F-tests', 'Tuned Model on 10 Variables Selected by MRMR'};
    res.Properties.RowNames = {'PCA Classifier', 'PLS Classifier'};
    
    %save the results
    writetable(res,'res2.csv','WriteRowNames',true);
end
%% Auxilary functions to the main function
%this function calculates model metrics for predicted and actual ys of a
%regression
function [RMSE,R2,bias] = model_metrics(yPred,yAct)
    RMSE = rmse(yPred,yAct);
    SSE = sum((yPred-yAct).^2);
    bias = SSE/size(yPred,1);
    SSR = sum((yAct-mean(yAct)).^2);
    R2 = 1 - SSE/SSR;
end

%this function calculates model metrics for predicted and actual ys of a
%classification
function [acc,sens] = model_metrics_classifier(yPred,yAct)
    perf = classperf(yAct,yPred);
    acc = perf.CorrectRate;
    sens = perf.Sensitivity;
end

%this function trains a PCA classifier and produces model metrics
function [predYtest, predYtrain, accuVal, sensVal, accuTest, sensTest,ncomp] = pca_classifier(xTrain,xTest,yTrain,yTest)
    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    %tune for the best number of components with 5 folds
    ncomp = pca_tuner_class(xTrain,yTrain,5);
    
    %get the PCAs
    [pcaTrain,pcaScoresTrain] = pca(xTrainScaled);

    %regress for PCAs and calculate beta
    betaPCR = regress(yTrain-mean(yTrain), pcaScoresTrain(:,1:ncomp));
        betaPCR = pcaTrain(:,1:ncomp)*betaPCR;
    betaPCR = [mean(yTrain) - mean(xTrainScaled)*betaPCR; betaPCR];
    
    %predict for training and test set based on calculated beta
    predYtrain = round([ones(size(xTrainScaled,1),1) xTrainScaled]*betaPCR,0);
    predYtest = round([ones(size(xTestScaled,1),1) xTestScaled]*betaPCR,0);

    %convert the results to binary
    predYtrain(predYtrain<0) = 0;
    predYtrain(predYtrain>1) = 1;
    predYtest(predYtest<0) = 0;
    predYtest(predYtest>1) = 1;
    
    %produce the metrics for the training and test set
    [accuVal,sensVal] = model_metrics_classifier(predYtrain,yTrain);
    [accuTest,sensTest] = model_metrics_classifier(predYtest,yTest);
end

%this function trains a PCA classifier and produces model metrics based on
%10 variables selected by MRMR

function [predYtest, predYtrain, accuVal, sensVal, accuTest, sensTest,ncomp] = pca_classifier_fsrmrmr(xTrain,xTest,yTrain,yTest)
    
    %calculate mrmrs and select the first 10 most important variables
    [idx,~] = fsrmrmr(xTrain,yTrain);
    xTrain = xTrain(:,idx(1:10));
    xTest = xTest(:,idx(1:10));
    
    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    %tune for the best number of components with 5 folds
    ncomp = pca_tuner_class(xTrain,yTrain,5);

    %get the PCAs    
    [pcaTrain,pcaScoresTrain] = pca(xTrainScaled);

    %calculate beta
    betaPCR = regress(yTrain-mean(yTrain), pcaScoresTrain(:,1:ncomp));
    betaPCR = pcaTrain(:,1:ncomp)*betaPCR;
    betaPCR = [mean(yTrain) - mean(xTrainScaled)*betaPCR; betaPCR];
    
    %predict for the training and test set based on calculated beta
    predYtrain = round([ones(size(xTrainScaled,1),1) xTrainScaled]*betaPCR,0);
    predYtest = round([ones(size(xTestScaled,1),1) xTestScaled]*betaPCR,0);

    %convert the predictions to binary
    predYtrain(predYtrain<0) = 0;
    predYtrain(predYtrain>1) = 1;
    predYtest(predYtest<0) = 0;
    predYtest(predYtest>1) = 1;

    %produce model metrics
    [accuVal,sensVal] = model_metrics_classifier(predYtrain,yTrain);
    [accuTest,sensTest] = model_metrics_classifier(predYtest,yTest);
end


%this function trains a PCA classifier and produces model metrics based on
%10 variables selected by FFS
function [predYtest, predYtrain, accuVal, sensVal, accuTest, sensTest,ncomp] = pca_classifier_fsrtest(xTrain,xTest,yTrain,yTest)

    %calculate FFS and select the first 10 most important variables   
    [idx,~] = fsrftest(xTrain,yTrain);
    xTrain = xTrain(:,idx(1:10));
    xTest = xTest(:,idx(1:10));

    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    %tune for the best number of components with 5 folds
    ncomp = pca_tuner_class(xTrain,yTrain,5);
    
    %get the PCAs       
    [pcaTrain,pcaScoresTrain] = pca(xTrainScaled);

    %calculate beta
    betaPCR = regress(yTrain-mean(yTrain), pcaScoresTrain(:,1:ncomp));
    betaPCR = pcaTrain(:,1:ncomp)*betaPCR;
    betaPCR = [mean(yTrain) - mean(xTrainScaled)*betaPCR; betaPCR];
    
    %predict for the training and test set based on calculated beta
    predYtrain = round([ones(size(xTrainScaled,1),1) xTrainScaled]*betaPCR,0);
    predYtest = round([ones(size(xTestScaled,1),1) xTestScaled]*betaPCR,0);

    %convert the predictions to binary
    predYtrain(predYtrain<0) = 0;
    predYtrain(predYtrain>1) = 1;
    predYtest(predYtest<0) = 0;
    predYtest(predYtest>1) = 1;

    %produce model metrics
    [accuVal,sensVal] = model_metrics_classifier(predYtrain,yTrain);
    [accuTest,sensTest] = model_metrics_classifier(predYtest,yTest);
end


%this function trains a PCA regression, similar to PCA classifier
function [predYtest, predYtrain, rmseVal, rmseTest, r2Val, r2Test, biasVal, biasTest,ncomp] = pca_regression(xTrain,xTest,yTrain,yTest)
    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    ncomp = pca_tuner(xTrain,yTrain,5);
    
    [pcaTrain,pcaScoresTrain] = pca(xTrainScaled);


    betaPCR = regress(yTrain-mean(yTrain), pcaScoresTrain(:,1:ncomp));

    betaPCR = pcaTrain(:,1:ncomp)*betaPCR;

    betaPCR = [mean(yTrain) - mean(xTrainScaled)*betaPCR; betaPCR];
    
    predYtrain = [ones(size(xTrainScaled,1),1) xTrainScaled]*betaPCR;
    predYtest = [ones(size(xTestScaled,1),1) xTestScaled]*betaPCR;

    [rmseVal,r2Val,biasVal] = model_metrics(predYtrain,yTrain);
    [rmseTest,r2Test,biasTest] = model_metrics(predYtest,yTest);
end


%this function tunes a PCA regression for number of components based on k
%folds
function ncomp = pca_tuner(xTrain,yTrain,k)
    %assume maximum number of components to test
    ncompTest = min((size(xTrain,1))/2,(size(xTrain,2))/2);
    rng(0)
    %create a cross-validation partitions
    c = cvpartition(size(xTrain,1),"KFold",k);
    %template for results
    res_comp = zeros(ncompTest,1);

    %iterate through each number of components and train and test for all
    %folds and calculate RMSE
    for i=1:ncompTest
        predFolds = zeros(0,2);
        for j=1:k
            xTrainK = xTrain(training(c,k),:);
            yTrainK = yTrain(training(c,k));
            xTestK = xTrain(test(c,k),:);
            yTestK = yTrain(test(c,k));

            [xTrainKScaled, xTestKScaled] = scale_train_test(xTrainK,xTestK);

            [pcaTrainK,pcaScoresTrainK] = pca(xTrainKScaled);

        
            betaPCR = regress(yTrainK-mean(yTrainK), pcaScoresTrainK(:,1:i));
        
            betaPCR = pcaTrainK(:,1:i)*betaPCR;
        
            betaPCR = [mean(yTrainK) - mean(xTrainKScaled)*betaPCR; betaPCR];

            predYtestK = [ones(size(xTestKScaled,1),1) xTestKScaled]*betaPCR;

            predFolds = [predFolds;[predYtestK,yTestK]];
        end
        res_comp(i) =  model_metrics(predFolds(:,1),predFolds(:,2));
    end

    %find the best number of components
    [~,ncomp_cand] = min(res_comp);
    ncomp_cand_new = ncomp_cand;
    while true
        if ncomp_cand_new == 1
            break;
        %to make sure unnecessary large number of components is not
        %selected, the minimum RMSE is compared to the n-1 components RMSE
        %and if they are only 1 percent different, n-1 is selected as the
        %new number of components. This step is repeated until the
        %difference is greater than 1 percent. 
        elseif ((res_comp(ncomp_cand_new)- res_comp(ncomp_cand_new-1))/res_comp(ncomp_cand))<0.01
            ncomp_cand_new = ncomp_cand_new - 1;
        else
            break;
        end
    end
    ncomp = ncomp_cand_new;
end


%this function tunes a PCA classifier for number of components based on k
%folds. Similar to previous tuner. 
function ncomp = pca_tuner_class(xTrain,yTrain,k)
    %determine the number of components to test for
    ncompTest = min((size(xTrain,1))/2,(size(xTrain,2))/2);
    rng(0)
    c = cvpartition(size(xTrain,1),"KFold",k);
    res_comp = zeros(ncompTest,1);

    for i=1:ncompTest
        predFolds = zeros(0,2);

        for j=1:k

            xTrainK = xTrain(training(c,k),:);
            yTrainK = yTrain(training(c,k));
            xTestK = xTrain(test(c,k),:);
            yTestK = yTrain(test(c,k));

            [xTrainKScaled, xTestKScaled] = scale_train_test(xTrainK,xTestK);

            [pcaTrainK,pcaScoresTrainK] = pca(xTrainKScaled);

        
            betaPCR = regress(yTrainK-mean(yTrainK), pcaScoresTrainK(:,1:i));
        
            betaPCR = pcaTrainK(:,1:i)*betaPCR;
        
            betaPCR = [mean(yTrainK) - mean(xTrainKScaled)*betaPCR; betaPCR];

            predYtestK = round([ones(size(xTestKScaled,1),1) xTestKScaled]*betaPCR,0);
            
            predYtestK(predYtestK<0) = 0;
            predYtestK(predYtestK>1) = 1;

            predFolds = [predFolds;[predYtestK,yTestK]];


        end
        res_comp(i) =  model_metrics_classifier(predFolds(:,1),predFolds(:,2));
    end

    [~,ncomp_cand] = max(res_comp);
    ncomp_cand_new = ncomp_cand;
    while true
        if ncomp_cand_new == 1
            break;
        elseif ((res_comp(ncomp_cand_new)- res_comp(ncomp_cand_new-1))/res_comp(ncomp_cand))<0.01
            ncomp_cand_new = ncomp_cand_new - 1;
        else
            break;
        end
    end
    ncomp = ncomp_cand_new;
end


%this function trains and tests a PLS classifier, similar to PCA classifier 
function [predYtest, predYtrain, accuVal, sensVal, accuTest, sensTest,ncomp] = pls_classifier(xTrain,xTest,yTrain,yTest)
    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    ncomp = pls_tuner_class(xTrain,yTrain,5);

    [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainScaled,yTrain,ncomp);

    predYtrain = round([ones(size(xTrainScaled,1),1) xTrainScaled]*beta,0);

    predYtest = round([ones(size(xTestScaled,1),1) xTestScaled]*beta,0);
    
    predYtrain(predYtrain<0) = 0;
    predYtrain(predYtrain>1) = 1;
    predYtest(predYtest<0) = 0;
    predYtest(predYtest>1) = 1;
    
    [accuVal,sensVal] = model_metrics_classifier(predYtrain,yTrain);
    [accuTest,sensTest] = model_metrics_classifier(predYtest,yTest);
end

%this function trains and tests a PLS classifier with fist 10 variables
%selected based on MRMR
function [predYtest, predYtrain, accuVal, sensVal, accuTest, sensTest,ncomp] = pls_classifier_fsrmrmr(xTrain,xTest,yTrain,yTest)

    %select the first 10 varibles
    [idx,~] = fsrmrmr(xTrain,yTrain);
    xTrain = xTrain(:,idx(1:10));
    xTest = xTest(:,idx(1:10));

    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    ncomp = pls_tuner_class(xTrain,yTrain,5);

    [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainScaled,yTrain,ncomp);

    predYtrain = round([ones(size(xTrainScaled,1),1) xTrainScaled]*beta,0);

    predYtest = round([ones(size(xTestScaled,1),1) xTestScaled]*beta,0);
    
    predYtrain(predYtrain<0) = 0;
    predYtrain(predYtrain>1) = 1;
    predYtest(predYtest<0) = 0;
    predYtest(predYtest>1) = 1;
    
    [accuVal,sensVal] = model_metrics_classifier(predYtrain,yTrain);
    [accuTest,sensTest] = model_metrics_classifier(predYtest,yTest);
end


%this function trains and tests a PLS classifier with fist 10 variables
%selected based on FFS
function [predYtest, predYtrain, accuVal, sensVal, accuTest, sensTest,ncomp] = pls_classifier_fsrtest(xTrain,xTest,yTrain,yTest)

    %select the first 10 varibles    
    [idx,~] = fsrftest(xTrain,yTrain);
    xTrain = xTrain(:,idx(1:10));
    xTest = xTest(:,idx(1:10));

    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    ncomp = pls_tuner_class(xTrain,yTrain,5);

    [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainScaled,yTrain,ncomp);

    predYtrain = round([ones(size(xTrainScaled,1),1) xTrainScaled]*beta,0);

    predYtest = round([ones(size(xTestScaled,1),1) xTestScaled]*beta,0);
    
    predYtrain(predYtrain<0) = 0;
    predYtrain(predYtrain>1) = 1;
    predYtest(predYtest<0) = 0;
    predYtest(predYtest>1) = 1;
    
    [accuVal,sensVal] = model_metrics_classifier(predYtrain,yTrain);
    [accuTest,sensTest] = model_metrics_classifier(predYtest,yTest);
end


%this function trains and tests a PLS regression 
function [predYtest, predYtrain, rmseVal, rmseTest, r2Val, r2Test, biasVal, biasTest,ncomp] = pls_regression(xTrain,xTest,yTrain,yTest)
    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    ncomp = pls_tuner(xTrain,yTrain,5);

    [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainScaled,yTrain,ncomp);

    predYtrain = [ones(size(xTrainScaled,1),1) xTrainScaled]*beta;

    predYtest = [ones(size(xTestScaled,1),1) xTestScaled]*beta;
    

    [rmseVal,r2Val,biasVal] = model_metrics(predYtrain,yTrain);
    [rmseTest,r2Test,biasTest] = model_metrics(predYtest,yTest);
end


%this function trains and tests a PLS classifier with fist 10 variables
%selected based on MRMR
function [predYtest, predYtrain, rmseVal, rmseTest, r2Val, r2Test, biasVal, biasTest,ncomp] = pls_regression_fsrmrmr(xTrain,xTest,yTrain,yTest)
    
    %select the first 10 variables
    idx = fsrmrmr(xTrain,yTrain);
    xTrain = xTrain(:,idx(1:10));
    xTest = xTest(:,idx(1:10));

    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    ncomp = pls_tuner(xTrain,yTrain,5);

    [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainScaled,yTrain,ncomp);

    predYtrain = [ones(size(xTrainScaled,1),1) xTrainScaled]*beta;

    predYtest = [ones(size(xTestScaled,1),1) xTestScaled]*beta;
    

    [rmseVal,r2Val,biasVal] = model_metrics(predYtrain,yTrain);
    [rmseTest,r2Test,biasTest] = model_metrics(predYtest,yTest);
end


%this function trains and tests a PLS classifier with fist 10 variables
%selected based on FFS

function [predYtest, predYtrain, rmseVal, rmseTest, r2Val, r2Test, biasVal, biasTest,ncomp] = pls_regression_fsrtest(xTrain,xTest,yTrain,yTest)
    %select the first 10 variables
    idx = fsrftest(xTrain,yTrain);
    xTrain = xTrain(:,idx(1:10));
    xTest = xTest(:,idx(1:10));

    %scale the training and test set
    [xTrainScaled, xTestScaled] = scale_train_test(xTrain,xTest);

    ncomp = pls_tuner(xTrain,yTrain,5);

    [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainScaled,yTrain,ncomp);

    predYtrain = [ones(size(xTrainScaled,1),1) xTrainScaled]*beta;

    predYtest = [ones(size(xTestScaled,1),1) xTestScaled]*beta;
    

    [rmseVal,r2Val,biasVal] = model_metrics(predYtrain,yTrain);
    [rmseTest,r2Test,biasTest] = model_metrics(predYtest,yTest);
end


%this function tunes a PLS regression for number of components based on k
%folds
function ncomp = pls_tuner(xTrain,yTrain,k)
    ncompTest = min((size(xTrain,1))/2,(size(xTrain,2))/2);
    rng(0)
    c = cvpartition(size(xTrain,1),"KFold",k);
    res_comp = zeros(ncompTest,1);

    for i=1:ncompTest
        predFolds = zeros(0,2);

        for j=1:k

            xTrainK = xTrain(training(c,k),:);
            yTrainK = yTrain(training(c,k));
            xTestK = xTrain(test(c,k),:);
            yTestK = yTrain(test(c,k));

            [xTrainKScaled, xTestKScaled] = scale_train_test(xTrainK,xTestK);

            [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainKScaled,yTrainK,i);

            predYtestK = [ones(size(xTestKScaled,1),1) xTestKScaled]*beta;

            predFolds = [predFolds;[predYtestK,yTestK]];
        end
        res_comp(i) =  model_metrics(predFolds(:,1),predFolds(:,2));
    end

    [~,ncomp_cand] = min(res_comp);
    ncomp_cand_new = ncomp_cand;
    while true
        if ncomp_cand_new == 1
            break;
        elseif (abs((res_comp(ncomp_cand_new)- res_comp(ncomp_cand_new-1))/res_comp(ncomp_cand)))<0.001
            ncomp_cand_new = ncomp_cand_new - 1;
        else
            break;
        end
    end
    ncomp = ncomp_cand_new;
end



%this function tunes a PLS classifier for number of components based on k
%folds

function ncomp = pls_tuner_class(xTrain,yTrain,k)
    ncompTest = min((size(xTrain,1))/2,(size(xTrain,2))/2);
    rng(0)
    c = cvpartition(size(xTrain,1),"KFold",k);
    res_comp = zeros(ncompTest,1);

    for i=1:ncompTest
        predFolds = zeros(0,2);

        for j=1:k

            xTrainK = xTrain(training(c,k),:);
            yTrainK = yTrain(training(c,k));
            xTestK = xTrain(test(c,k),:);
            yTestK = yTrain(test(c,k));

            [xTrainKScaled, xTestKScaled] = scale_train_test(xTrainK,xTestK);

            [XL,yl,XS,YS,beta,PCTVAR] = plsregress(xTrainKScaled,yTrainK,i);

            predYtestK = round([ones(size(xTestKScaled,1),1) xTestKScaled]*beta,0);
            
            predYtestK(predYtestK<0) = 0;
            predYtestK(predYtestK>1) = 1;

            predFolds = [predFolds;[predYtestK,yTestK]];
        end
        res_comp(i) =  model_metrics_classifier(predFolds(:,1),predFolds(:,2));
    end

    [~,ncomp_cand] = min(res_comp);
    ncomp_cand_new = ncomp_cand;
    while true
        if ncomp_cand_new == 1
            break;
        elseif (abs((res_comp(ncomp_cand_new)- res_comp(ncomp_cand_new-1))/res_comp(ncomp_cand)))<0.001
            ncomp_cand_new = ncomp_cand_new - 1;
        else
            break;
        end
    end
    ncomp = ncomp_cand_new;
end

%This function takes a training set, mean center it and also mean center
%the test set based on the training set mean center parameters
function [xTrainScaled,xTestScaled] = scale_train_test(xTrain,xTest)
    %scale the training set set
    [xTrainScaled,C,S] = normalize(xTrain,1,'center','mean','scale');
    xTestScaled = (xTest-C)./S; 
end