# H2O Random Forest Starter Script
# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example
# Forked from Random Forest Example by Michael Pawlus

# Use data.table and H2O to create random forest predictions for the entire
#   training set.
# This is a starter script so it mostly uses the provided fields, as is.
# A log transform has been applied to the Sales, year and month are used,
#  and the Store ID is used as a factor.
# Vesion 18 does not remove the 0 readings, which does help, since we are not judged
#  on those entries. The model found "Open" to be the most important feature,
#  which makes sense, since Sales are 0 when the store is not open. However, 
#  since observations with 0 Sales are discarded by Kaggle upon judging, it
#  is a better strategy to remove them, as Michael's original script does.
#   And version 19/20 started removing the 0 records for 0.004 improvement. 
# To make the benchmark a little more competitive, this has more and deeper 
#  trees than the original. If you want to see it run faster, you can lower those
#  settings while you work through some other parts of the model, and increase them
#  later.
# Also, the h2o.gbm() has many shared parameters, so you can try that out as well,
#  and these parameters will work (though you probably don't want depth 30 for GBM).
# And to add variety, you can try out h2o.glm(), a regularized GLM, or 
#  h2o.deeplearning(), for deep neural networks. This code will work for either with
#  the exception of the ntrees, max_depth, and nbins_cats, which are decision tree
#  parameters.
# Good luck!

library(data.table)  
library(h2o)

cat("reading the train and test data (with data.table) \n")
train <- fread("../input/train.csv",stringsAsFactors = T)
test  <- fread("../input/test.csv",stringsAsFactors = T)
store <- fread("../input/store.csv",stringsAsFactors = T)
train <- train[Sales > 0,]  ## We are not judged on 0 sales records in test set
    ## See Scripts discussion from 10/8 for more explanation.
train <- merge(train,store,by="Store")
test <- merge(test,store,by="Store")

cat("train data column names and details\n")
summary(train)
cat("test data column names and details\n")
summary(test)

## more care should be taken to ensure the dates of test can be projected from train
## decision trees do not project well, so you will want to have some strategy here, if using the dates
train[,Date:=as.Date(Date)]
test[,Date:=as.Date(Date)]

# seperating out the elements of the date column for the train set
train[,month:=as.integer(format(Date, "%m"))]
train[,year:=as.integer(format(Date, "%y"))]
train[,Store:=as.factor(as.numeric(Store))]

test[,month:=as.integer(format(Date, "%m"))]
test[,year:=as.integer(format(Date, "%y"))]
test[,Store:=as.factor(as.numeric(Store))]

## log transformation to not be as sensitive to high sales
## decent rule of thumb: 
##     if the data spans an order of magnitude, consider a log transform
train[,logSales:=log1p(Sales)]

## Use H2O's random forest
## Start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='6G')
## Load data into cluster from R
trainHex<-as.h2o(train)
## Set up variable to use all features other than those specified here
features<-colnames(train)[!(colnames(train) %in% c("Id","Date","Sales","logSales","Customers"))]
## Train a random forest using all default parameters
rfHex <- h2o.randomForest(x=features,
                          y="logSales", 
                          ntrees = 200,
                          max_depth = 30,
                          nbins_cats = 1115, ## allow it to fit store ID
                          training_frame=trainHex)

summary(rfHex)
cat("Predicting Sales\n")
## Load test data into cluster from R
testHex<-as.h2o(test)

## Get predictions out; predicts in H2O, as.data.frame gets them into R
predictions<-as.data.frame(h2o.predict(rfHex,testHex))
## Return the predictions to the original scale of the Sales data
pred <- expm1(predictions[,1])
summary(pred)
submission <- data.frame(Id=test$Id, Sales=pred)

cat("saving the submission file\n")
write.csv(submission, "h2o_rf.csv",row.names=F)

####################################################################################
## Appendix: h2o.randomForest API
####################################################################################
##h2o.randomForest(x, y, training_frame, model_id, validation_frame, checkpoint,
##  mtries = -1, sample_rate = 0.632, build_tree_one_node = FALSE,
##  ntrees = 50, max_depth = 20, min_rows = 1, nbins = 20,
##  nbins_cats = 1024, binomial_double_trees = FALSE,
##  balance_classes = FALSE, max_after_balance_size = 5, seed,
##  offset_column = NULL, weights_column = NULL, nfolds = 0,
##  fold_column = NULL, fold_assignment = c("AUTO", "Random", "Modulo"),
##  keep_cross_validation_predictions = FALSE, ...)
## Arguments

## x	
## A vector containing the names or indices of the predictor variables to use in building the GBM model.

## y 
## The name or index of the response variable. If the data does not contain a header, this is the column index number starting at 1, and increasing from left to right. (The response must be either an integer or a categorical variable).

## training_frame    
## An H2OFrame object containing the variables in the model.

## model_id    
## (Optional) The unique id assigned to the resulting model. If none is given, an id will automatically be generated.

## validation_frame	 
## An H2OFrame object containing the variables in the model.

## checkpoint  
## "Model checkpoint (either key or H2ODeepLearningModel) to resume training with."

## mtries 
## Number of variables randomly sampled as candidates at each split. If set to -1, defaults to sqrtp for classification, and p/3 for regression, where p is the number of predictors.

## sample_rate	       
## Sample rate, from 0 to 1.0.   (edit: row sampling, per tree)

## build_tree_one_node 
## Run on one node only; no network overhead but fewer cpus used. Suitable for small datasets.

## ntrees 
## A nonnegative integer that determines the number of trees to grow.

## max_depth	 
## Maximum depth to grow the tree.

## min_rows	 
## Minimum number of rows to assign to teminal nodes.

## nbins   
## For numerical columns (real/int), build a histogram of this many bins, then split at the best point.

## nbins_cats	 
## For categorical columns (enum), build a histogram of this many bins, then split at the best point. Higher values can lead to more overfitting.

## binomial_double_trees   
## For binary classification: Build 2x as many trees (one per class) - can lead to higher accuracy.

## balance_classes	      
## logical, indicates whether or not to balance training data class counts via over/under-sampling (for imbalanced data)

## max_after_balance_size     
## Maximum relative size of the training data after balancing class counts (can be less than 1.0)

## seed	   
## Seed for random numbers (affects sampling) - Note: only reproducible when running single threaded

## offset_column   
## Specify the offset column.

## weights_column     
## Specify the weights column.

## nfolds  
## (Optional) Number of folds for cross-validation. If nfolds >= 2, then validation must remain empty.

## fold_column	     
## (Optional) Column with cross-validation fold index assignment per observation

## fold_assignment   
## Cross-validation fold assignment scheme, if fold_column is not specified Must be "AUTO", "Random" or "Modulo"

## keep_cross_validation_predictions	    
## Whether to keep the predictions of the cross-validation models
                