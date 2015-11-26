# ---
#   title: "Rossmann_H2O_Random_Forest_model_3.2"
# author: "David Comfort"
# date: "November 18, 2015"
# output: html_document
# ---
#   
#   Forecast sales using store, promotion, and competitor data
# 
# Rossmann Store Sales
# 
# Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.
# 
# In their first Kaggle competition, Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany. Reliable sales forecasts enable store managers to create effective staff schedules that increase productivity and motivation. By helping Rossmann create a robust prediction model, you will help store managers stay focused on what’s most important to them: their customers and their teams!
#   
#   H2O Random Forest Starter Script
# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example
# Forked from Random Forest Example by Michael Pawlus
# 
# Use data.table and H2O to create random forest predictions for the entire training set. This is a starter script so it mostly uses the provided fields, as is.
# 
# * A log transform has been applied to the Sales, year and month are used, and the Store ID is used as a factor.
# * Version 18 does not remove the 0 readings, which does help, since we are not judged on those entries. 
# * The model found "Open" to be the most important feature, which makes sense, since Sales are 0 when the store is not open. However, since observations with 0 Sales are discarded by Kaggle upon judging, it is a better strategy to remove them, as Michael's original script does.
# * And version 19/20 started removing the 0 records for 0.004 improvement. 
# * To make the benchmark a little more competitive, this has more and deeper trees than the original. If you want to see it run faster, you can lower those settings while you work through some other parts of the model, and increase them later.
# * Also, the h2o.gbm() has many shared parameters, so you can try that out as well,  and these parameters will work (though you probably don't want depth 30 for GBM).
# * And to add variety, you can try out h2o.glm(), a regularized GLM, or h2o.deeplearning(), for deep neural networks. * * This code will work for either with the exception of the ntrees, max_depth, and nbins_cats, which are decision tree  parameters.
# * Good luck!
  

library(data.table)  
library(h2o)

# localH2O = h2o.init()
localH2O = h2o.init(nthreads = -1, max_mem_size = '6g', assertion = FALSE)
h2o.removeAll() # Clean slate - just in case the cluster was already running
cat("reading the train and test data (with data.table) \n")
train <- fread("data/train_states_R_v5.csv",stringsAsFactors = T)
# train <- fread("data/train.csv",stringsAsFactors = T)
str(train)
test  <- fread("data/test_states_R_v5.csv",stringsAsFactors = T)
# test  <- fread("data/test.csv",stringsAsFactors = T)
str(test)
store <- fread("data/store.csv",stringsAsFactors = T)
str(store)
## We are not judged on 0 sales records in test set
train <- train[Sales > 0,]  ## We are not judged on 0 sales records in test set


# impute Competition Values 
store$CompetitionOpenSinceYear[is.na(store$CompetitionOpenSinceYear)] <- 1990 # Dealing with NA and outlayers
store$CompetitionOpenSinceMonth[is.na(store$CompetitionOpenSinceMonth)] <- 1 # Dealing with NA
store$CompetitionDistance[is.na(store$CompetitionDistance)] <- 75000 # Dealing with NA

# Competition strength
# store$CompetitionStrength <- cut(store$CompetitionDistance, breaks=c(0, 1500, 6000, 12000, 20000, Inf), labels=FALSE) # 15 min, 1/2/3 hours (or walking and 10/20/30 min driving)

# train$CompetitionStrength <- as.factor(train$CompetitionStrength)
# test$CompetitionStrength <- as.factor(test$CompetitionStrength)



## Merge Train and Test with Store
train <- merge(train,store,by="Store")
str(train)
test <- merge(test,store,by="Store")
str(test)


## Summarize the Data:
  
  
cat("train data column names and details\n")
summary(train)
cat("test data column names and details\n")
summary(test)


# * More care should be taken to ensure the dates of test can be projected from train
# * Decision trees do not project well, so you will want to have some strategy here, if using the dates

train[,Date:=as.Date(Date)]
test[,Date:=as.Date(Date)]

# Competition feature
train$Competition <- (sqrt(max(train$CompetitionDistance, na.rm = TRUE)-train$CompetitionDistance))*
  (((train$year - train$CompetitionOpenSinceYear) * 12) - (train$CompetitionOpenSinceMonth-train$month))

test$Competition <- (sqrt(max(test$CompetitionDistance, na.rm = TRUE)-test$CompetitionDistance))*
  (((test$year - test$CompetitionOpenSinceYear) * 12) - (test$CompetitionOpenSinceMonth-test$month))

# * change to factors
train$PromoFirstDate <- as.factor(train$PromoFirstDate)
test$PromoFirstDate <- as.factor(test$PromoFirstDate)

train$Promo <- as.factor(train$Promo)
test$Promo <- as.factor(test$Promo)

train$DayOfWeek <- as.factor(train$DayOfWeek)
test$DayOfWeek <- as.factor(test$DayOfWeek)

train$day <- as.factor(train$day)
test$day <- as.factor(test$day)

train$CompetitionOpenSinceMonth <- as.factor(train$CompetitionOpenSinceMonth)
test$CompetitionOpenSinceMonth <- as.factor(test$CompetitionOpenSinceMonth)

train$weeknum <- as.factor(train$weeknum)
test$weeknum <- as.factor(test$weeknum)

train$day_of_year <- as.factor(train$day_of_year)
test$day_of_year <- as.factor(test$day_of_year)


# * Separating out the elements of the date column for the train set
train[,Store:=as.factor(as.numeric(Store))]
test[,Store:=as.factor(as.numeric(Store))]

# * log transformation to not be as sensitive to high sales
# *  decent rule of thumb:if the data spans an order of magnitude, consider a log transform

# train[,logSales:=log1p(Sales)]
train$logSales <- log1p(train$Sales)

## Use H2O's random forest
## Start cluster with all available threads
# h2o.init(nthreads=-1,max_mem_size='6G')

## Load data into cluster from R
trainHex<-as.h2o(train)


## Set up variable to use all features other than those specified here
features<-colnames(train)[!(colnames(train) %in% c("Id","Date","Sales","logSales",
                                                   "Customers", "is_quarter_start", 
                                                   "is_month_start", "is_quarter_end",
                                                   "Pop_per_store", "pop_den_per_store",
                                                   "quarter", "is_month_end"
                                                   ))]

## Train a random forest using all default parameters
rfHex <- h2o.randomForest(x=features,
                          y="logSales", 
                          ntrees = 100, 
                          max_depth = 30,  
                          nbins_cats = 1115, ## allow it to fit store ID
                          training_frame=trainHex)

summary(rfHex)
cat("Predicting Sales\n")

varimps = data.frame(h2o.varimp(rfHex))
varimps
## Load test data into cluster from R
testHex<-as.h2o(test)

## Get predictions out; predicts in H2O, as.data.frame gets them into R
predictions<-as.data.frame(h2o.predict(rfHex,testHex))

## Return the predictions to the original scale of the Sales data
pred <- expm1(predictions[,1])
summary(pred)
submission <- data.frame(Id=test$Id, Sales=pred)

cat("saving the submission file\n")
write.csv(submission, "data/h2o_100trees_v4.csv",row.names=F)


# ## Appendix: h2o.randomForest API
# * h2o.randomForest(x, y, training_frame, model_id, validation_frame, checkpoint,
#                    * mtries = -1, sample_rate = 0.632, build_tree_one_node = FALSE,
#                    * ntrees = 50, max_depth = 20, min_rows = 1, nbins = 20,
#                    * nbins_cats = 1024, binomial_double_trees = FALSE,
#                    * balance_classes = FALSE, max_after_balance_size = 5, seed,
#                    * offset_column = NULL, weights_column = NULL, nfolds = 0,
#                    * fold_column = NULL, fold_assignment = c("AUTO", "Random", "Modulo"),
#                    * keep_cross_validation_predictions = FALSE, ...)
# * Arguments
# 
# * x	
# * A vector containing the names or indices of the predictor variables to use in building the GBM model.
# 
# * y	
# * The name or index of the response variable. If the data does not contain a header, this is the column index number starting at 1, and increasing from left to right. (The response must be either an integer or a categorical variable).
# 
# * training_frame	
# * An H2OFrame object containing the variables in the model.
# 
# * model_id	
# * (Optional) The unique id assigned to the resulting model. If none is given, an id will automatically be generated.
# 
# * validation_frame	
# * An H2OFrame object containing the variables in the model.
# 
# * checkpoint	
# * "Model checkpoint (either key or H2ODeepLearningModel) to resume training with."
# 
# * mtries	
# * Number of variables randomly sampled as candidates at each split. If set to -1, defaults to sqrtp for classification, and p/3 for regression, where p is the number of predictors.
# 
# * sample_rate	
# * Sample rate, from 0 to 1.0.   (edit: row sampling, per tree)
# 
# * build_tree_one_node	
# * Run on one node only; no network overhead but fewer cpus used. Suitable for small datasets.
# 
# * ntrees	
# * A nonnegative integer that determines the number of trees to grow.
# 
# * max_depth	
# * Maximum depth to grow the tree.
# 
# * min_rows	
# * Minimum number of rows to assign to teminal nodes.
# 
# * nbins	
# * For numerical columns (real/int), build a histogram of this many bins, then split at the best point.
# 
# * nbins_cats	
# * For categorical columns (enum), build a histogram of this many bins, then split at the best point. Higher values can lead to more overfitting.
# 
# * binomial_double_trees	
# * For binary classification: Build 2x as many trees (one per class) - can lead to higher accuracy.
# 
# * balance_classes	
# * logical, indicates whether or not to balance training data class counts via over/under-sampling (for imbalanced data)
# 
# * max_after_balance_size	
# * Maximum relative size of the training data after balancing class counts (can be less than 1.0)
# 
# * seed	
# * Seed for random numbers (affects sampling) - Note: only reproducible when running single threaded
# 
# * offset_column	
# * Specify the offset column.
# 
# * weights_column	
# * Specify the weights column.
# 
# * nfolds	
# * (Optional) Number of folds for cross-validation. If nfolds >= 2, then validation must remain empty.
# 
# * fold_column	
# * (Optional) Column with cross-validation fold index assignment per observation
# 
# * fold_assignment	
# * Cross-validation fold assignment scheme, if fold_column is not specified Must be "AUTO", "Random" or "Modulo"
# 
# * keep_cross_validation_predictions	
# * Whether to keep the predictions of the cross-validation models

