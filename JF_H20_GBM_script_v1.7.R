# H2O GBM script version 1
library(caret)
library(plyr)
library(dplyr)
library(data.table)  
library(h2o)

cat("reading the train and test data (with data.table) \n")
train0 <- fread("../data/train3.csv",stringsAsFactors = T)
test  <- fread("../data/test3.csv",stringsAsFactors = T)
store <- fread("./input/store.csv",stringsAsFactors = T)
train0 <- train0[Sales > 0,]  ## We are not judged on 0 sales records in test set


train0 <- merge(train0,store,by="Store")
test <- merge(test,store,by="Store")


## more care should be taken to ensure the dates of test can be projected from train
## decision trees do not project well, so you will want to have some strategy here, if using the dates
train0[,Date:=as.Date(Date)]
test[,Date:=as.Date(Date)]

# competition feature
train0$Competition <- (sqrt(max(train0$CompetitionDistance, na.rm = TRUE)-train0$CompetitionDistance))*
  (((train0$year - train0$CompetitionOpenSinceYear) * 12) - (train0$CompetitionOpenSinceMonth-train0$month))

test$Competition <- (sqrt(max(test$CompetitionDistance, na.rm = TRUE)-test$CompetitionDistance))*
  (((test$year - test$CompetitionOpenSinceYear) * 12) - (test$CompetitionOpenSinceMonth-test$month))

## log transformation to not be as sensitive to high sales
## decent rule of thumb: 
##     if the data spans an order of magnitude, consider a log transform
train0[,logSales:=log1p(Sales)]
train  = train0
train_A = train0[day_of_year %% 2 == 0,]
train_B = train0[day_of_year %% 2 == 1,]
dim(valid)
dim(train)
unique(train0$day_of_year %% 2)

# Set appropriate variables to factors
for (j in c("Store", "DayOfWeek", "Promo", 
            "year", "month", "day", "PromoFirstDate",
#            "day_of_year", "week_of_year", "DayBeforeClosed", "DayAfterClosed",
            "State", "PromoSecondDate", 
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2", "Promo2SinceWeek", "Promo2SinceYear")) {
  train[[j]]   <- as.factor(train[[j]])
  train_A[[j]] <- as.factor(train_A[[j]])
  train_B[[j]] <- as.factor(train_B[[j]])
  test[[j]]    <- as.factor(test[[j]])
}


## Useful functions:
rmse = function(predictions, targets) {
  return(((predictions - targets)/targets) ** 2)
}

sumup = function(model, trainHex, train) {
  train_pred = as.data.frame(h2o.predict(model,trainHex))
  train_pred <- expm1(train_pred[,1])
  train$pred = train_pred
  train$rmse = rmse(train_pred, train$Sales)
  train2 = filter(train, month %in% c(8,9))
  total_rmse = sqrt(sum(train$rmse)/nrow(train))
  print("Total RMSE:")
  print(total_rmse)
  partial_rmse = sqrt(sum(train2$rmse)/nrow(train2))
  print("RMSE on Aug/Sep:")
  print(partial_rmse)
  temp = as.data.frame(rbind(summary(train_pred), summary(train$Sales), summary(train2$pred), summary(train2$Sales)))
  temp$sd = c(round(sd(train_pred)), round(sd(train$Sales)), round(sd(train2$pred)), round(sd(train2$Sales)))
  print("Stats of predictions vs. actual:")
  print(temp)
}


## Use H2O's random forest
## Start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='5G', assertion = FALSE)

## create validation and training set
trainHex   <- as.h2o(train)
trainHex_A <- as.h2o(train_A)
trainHex_B <- as.h2o(train_B)
testHex    <- as.h2o(test)

## Load data into cluster from R
features   = read.csv('./H2O_submits/h2o_GBM_20_03_800_top100_varimp.csv')
features   = features$variable
features   = as.character(features)

####################################################################################
gbmHex_full <- h2o.gbm( x               = features,
                        y               = "logSales",
                        training_frame  = trainHex,
                        model_id        = "introGBM",
                        nbins_cats      = 1115,
                        sample_rate     = 0.5,
                        col_sample_rate = 0.5,
                        max_depth       = 12,
                        learn_rate      = 0.5,
                        seed            = 12345678,
                        ntrees          = 200)

(varimps = data.frame(h2o.varimp(gbmHex_full)))
write.csv(varimps, "./H2O_submits/H2O_GBM_12_05_200_FULL_varimp.csv",row.names=F)
h2o.saveModel(gbmHex_full, path = '/Users/jfdarre/Documents/NYCDS/Project4/H2O_GBM_12_05_200_FULL', force = FALSE)

cat("Predicting Sales\n")
test_pred_full    <- as.data.frame(h2o.predict(gbmHex_full,testHex))
test_pred_full    <- expm1(test_pred_full[,1])
submit_full       <- data.frame(Id=test$Id, Sales=test_pred_full)
cat("saving the submission file\n")
write.csv(submit_CV, "./H2O_submits/H2O_GBM_12_05_200_full",row.names=F)

train_pred_full   <- as.data.frame(h2o.predict(gbmHex_full,trainHex))
train_pred_full   <- expm1(train_pred_full[,1])
train_pred_full   <- data.frame(Id=train$Id, Sales=train_pred_full)
write.csv(train_pred_full, "./H2O_submits/H2O_GBM_12_05_200_full_train.csv",row.names=F)

####################################################################################
gbmHex_A    <- h2o.gbm( x               = features,
                        y               = "logSales",
                        training_frame  = trainHex,
                        model_id        = "introGBM",
                        nbins_cats      = 1115,
                        sample_rate     = 0.5,
                        col_sample_rate = 0.5,
                        max_depth       = 12,
                        learn_rate      = 0.5,
                        seed            = 12345678,
                        ntrees          = 200)

(varimps = data.frame(h2o.varimp(gbmHex_A)))
write.csv(varimps, "./H2O_submits/H2O_GBM_12_05_200_A_varimp.csv",row.names=F)
h2o.saveModel(gbmHex_A, path = '/Users/jfdarre/Documents/NYCDS/Project4/H2O_GBM_12_05_200_A', force = FALSE)

cat("Predicting Sales\n")
test_pred_A    <- as.data.frame(h2o.predict(gbmHex_A,testHex))
test_pred_A    <- expm1(test_pred_A[,1])
submit_A       <- data.frame(Id=test$Id, Sales=test_pred_A)
cat("saving the submission file\n")
write.csv(submit_CV, "./H2O_submits/H2O_GBM_12_05_200_A",row.names=F)

train_pred_A   <- as.data.frame(h2o.predict(gbmHex_A,trainHex_B))
train_pred_A   <- expm1(train_pred_A[,1])
train_pred_A   <- data.frame(Id=train_B$Id, Sales=train_pred_A)
write.csv(train_pred_A, "./H2O_submits/H2O_GBM_12_05_200_A_train.csv",row.names=F)

####################################################################################
gbmHex_B    <- h2o.gbm( x               = features,
                        y               = "logSales",
                        training_frame  = trainHex,
                        model_id        = "introGBM",
                        nbins_cats      = 1115,
                        sample_rate     = 0.5,
                        col_sample_rate = 0.5,
                        max_depth       = 12,
                        learn_rate      = 0.5,
                        seed            = 12345678,
                        ntrees          = 200)

(varimps = data.frame(h2o.varimp(gbmHex_B)))
write.csv(varimps, "./H2O_submits/H2O_GBM_12_05_200_B_varimp.csv",row.names=F)
h2o.saveModel(gbmHex_B, path = '/Users/jfdarre/Documents/NYCDS/Project4/H2O_GBM_12_05_200_B', force = FALSE)

cat("Predicting Sales\n")
test_pred_B    <- as.data.frame(h2o.predict(gbmHex_B,testHex))
test_pred_B    <- expm1(test_pred_B[,1])
submit_B       <- data.frame(Id=test$Id, Sales=test_pred_B)
cat("saving the submission file\n")
write.csv(submit_CV, "./H2O_submits/H2O_GBM_12_05_200_B",row.names=F)

train_pred_B   <- as.data.frame(h2o.predict(gbmHex_B,trainHex_A))
train_pred_B   <- expm1(train_pred_B[,1])
train_pred_B   <- data.frame(Id=train_A$Id, Sales=train_pred_B)
write.csv(train_pred_B, "./H2O_submits/H2O_GBM_12_05_200_B_train.csv",row.names=F)

####################################################################################
e = seq(1,100,2)
f = seq(2,100,2)

####################################################################################
feats_e = features[e]

gbmHex_e <- h2o.gbm(x               = feats_e,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 12,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 200,
                    validation_frame = validHex)

summary(gbmHex_e)
h2o.saveModel(gbmHex_e, path = '../H2O_models_GBM_15_05_400_CV_e', force = FALSE)

cat("Predicting Sales\n")
pred_e       <- as.data.frame(h2o.predict(gbmHex_e, testHex))
pred_e       <- expm1(pred_e[,1])
submit_e     <- data.frame(Id = test$Id, Sales = pred_e)
cat("saving the submission file\n")
write.csv(submit_e, "./H2O_submits/H2O_models_GBM_15_05_400_CV_e.csv",row.names=F)

train_pred_e   <- as.data.frame(h2o.predict(gbmHex_e, trainHex))
train_pred_e   <- expm1(train_pred_e[,1])
train_pred_e   <- data.frame(Id = train$Id, Sales = train_pred_e)
write.csv(train_pred_e, "./H2O_submits/H2O_models_GBM_15_05_400_CV_e.csv",row.names=F)


####################################################################################
feats_f = features[f]

gbmHex_f <- h2o.gbm(x               = feats_f,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 12,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 200,
                    validation_frame = validHex)

summary(gbmHex_f)
h2o.saveModel(gbmHex_f, path = '../H2O_models_GBM_15_05_400_CV_f', force = FALSE)

cat("Predicting Sales\n")
pred_f       <- as.data.frame(h2o.predict(gbmHex_f, testHex))
pred_f       <- expm1(pred_f[,1])
submit_f     <- data.frame(Id = test$Id, Sales = pred_f)
cat("saving the submission file\n")
write.csv(submit_f, "./H2O_submits/H2O_models_GBM_15_05_400_CV_f.csv",row.names=F)

train_pred_f   <- as.data.frame(h2o.predict(gbmHex_f, trainHex))
train_pred_f   <- expm1(train_pred_f[,1])
train_pred_f   <- data.frame(Id = train$Id, Sales = train_pred_f)
write.csv(train_pred_f, "./H2O_submits/H2O_models_GBM_15_05_400_CV_f.csv",row.names=F)

####################################################################################
train_B$trainAB = log1p(train_pred_A$Sales)
train_A$trainAB = log1p(train_pred_B$Sales)
test$trainAB    = log1p(test_pred_full)
train_new       = rbind(train_A,train_B)
train_newHex    = as.h2o(train_new)


features        = c(features, "trainAB")

gbmHex_ensb <- h2o.gbm( x               = features,
                        y               = "logSales",
                        training_frame  = trainHex,
                        model_id        = "introGBM",
                        nbins_cats      = 1115,
                        sample_rate     = 0.5,
                        col_sample_rate = 0.5,
                        max_depth       = 12,
                        learn_rate      = 0.5,
                        seed            = 12345678,
                        ntrees          = 200)


