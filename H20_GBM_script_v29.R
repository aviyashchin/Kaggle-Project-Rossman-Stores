# H2O GBM script version 1
# v24 re-run same conditions as best scoring Kaggle GBM submission
# v25
# 25
# 26
# v27 re-run best conditions with nfolds=5
library(caret)
library(data.table)  
library(h2o)
library(plyr)

## Use H2O 
## Start cluster with all available threads
# use h2o.shutdown() if changing parameters below
h2o.init(nthreads=-1,max_mem_size='8G', assertion = FALSE)


cat("reading the train and test data (with data.table) \n")
train <- fread("../../KaggleProject/data/train_states_R_v8.csv",stringsAsFactors = T)
test  <- fread("../../KaggleProject/data/test_states_R_v8.csv",stringsAsFactors = T)
store <- fread("../../input/store.csv",stringsAsFactors = T)
train <- train[Sales > 0,]  ## We are not judged on 0 sales records in test set


## create stratified folds for cross-validation
# folds <- createFolds(factor(store$Store), k = 10, list = FALSE)
# store$fold <- folds
# ddply(store, 'fold', summarise, prop=mean(store$fold)/10)

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

# competition feature
train$Competition <- (sqrt(max(train$CompetitionDistance, na.rm = TRUE)-train$CompetitionDistance))*
  (((train$year - train$CompetitionOpenSinceYear) * 12) - (train$CompetitionOpenSinceMonth-train$month))

test$Competition <- (sqrt(max(test$CompetitionDistance, na.rm = TRUE)-test$CompetitionDistance))*
  (((test$year - test$CompetitionOpenSinceYear) * 12) - (test$CompetitionOpenSinceMonth-test$month))

# set Store to factor
train[,Store:=as.factor(as.numeric(Store))]
test[,Store:=as.factor(as.numeric(Store))]

str(train)
str(test)

# # Set appropriate variables to factors
# for (j in c("DayOfWeek", "Promo", "SchoolHoliday",
#             # "year", "month", 
#             "day", "day_of_year", "week_of_year", 
#             "PromoFirstDate","PromoSecondDate", 
#             "DayBeforeClosed", "DayAfterClosed",
#             "SundayStore", "DayBeforeRefurb", "DayAfterRefurb", "DaysBeforeRefurb", "DaysAfterRefurb",
#             "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
#             "Promo2", "Promo2SinceWeek", "Promo2SinceYear"
# )) {
#   train[[j]] <- as.factor(train[[j]])
#   test[[j]]  <- as.factor(test[[j]])
# }


## log transformation to not be as sensitive to high sales
## decent rule of thumb: 
##     if the data spans an order of magnitude, consider a log transform
train[,logSales:=log1p(Sales)]
str(train)
str(test)


# create validation and training set
trainHex<-as.h2o(train[year <2015 | month <6,],destination_frame = "trainHex")
validHex<-as.h2o(train[year == 2015 & month >= 6,],destination_frame = "validHex")
dim(trainHex); dim(validHex)

## Load data into cluster from R
### comment out below line if using validation set ####
# trainHex<-as.h2o(train)

################################################
features<-names(train)[!(names(train) %in% c("Id","Date","Sales","logSales", "Customers", "Closed", "fold"))]
features
####################################################################################
gbmHex <- h2o.gbm(x=features,
                  y="logSales",
                  training_frame=trainHex,
                  model_id="introGBM",
                  nbins_cats=1115,
                  sample_rate = 0.5,
                  col_sample_rate = 0.5,
                  max_depth = 15,
                  learn_rate=0.05,
                  seed = 12345678, #Seed for random numbers (affects sampling) - Note: only reproducible when running single threaded
                  ntrees = 200,
#                  fold_column="fold",
                  validation_frame=validHex) # validation set

summary(gbmHex)
gbmHex@parameters
varimps = data.frame(h2o.varimp(gbmHex))
varimps


####
# re-train on whole training set
trainHex<-as.h2o(train)

gbmHex <- h2o.gbm(x=features,
                  y="logSales",
                  training_frame=trainHex,
                  model_id="introGBM",
                  nbins_cats=1115,
                  sample_rate = 0.5,
                  col_sample_rate = 0.5,
                  max_depth = 15,
                  learn_rate=0.05,
                  seed = 12345678, #Seed for random numbers (affects sampling) - Note: only reproducible when running single threaded
#                  fold_column="fold",
                  ntrees = 200)

summary(gbmHex)
gbmHex@parameters
varimps = data.frame(h2o.varimp(gbmHex))
varimps


cat("Predicting Sales\n")
## Load test data into cluster from R
testHex<-as.h2o(test)

## Get predictions out; predicts in H2O, as.data.frame gets them into R
predictions<-as.data.frame(h2o.predict(gbmHex,testHex))
## Return the predictions to the original scale of the Sales data
pred <- expm1(predictions[,1])
summary(pred)
submission <- data.frame(Id=test$Id, Sales=pred)
cat("saving the submission file\n")
write.csv(submission, "../../data/H2O_GBM_v29.csv",row.names=F)

