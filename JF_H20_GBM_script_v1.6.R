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
valid = train0[year == 2015 & month >= 6,]
train = train0[year <  2015 | month <  6,]
dim(valid)
dim(train)


# Set appropriate variables to factors
for (j in c("Store", "DayOfWeek", "Promo", 
            "year", "month", "day", "PromoFirstDate",
#            "day_of_year", "week_of_year", "DayBeforeClosed", "DayAfterClosed",
            "State", "PromoSecondDate", 
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2", "Promo2SinceWeek", "Promo2SinceYear")) {
  train[[j]] <- as.factor(train[[j]])
  valid[[j]] <- as.factor(valid[[j]])
  test[[j]]  <- as.factor(test[[j]])
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
trainHex <- as.h2o(train)
validHex <- as.h2o(valid)
testHex  <- as.h2o(test)

## Load data into cluster from R
features   = read.csv('./H2O_submits/h2o_GBM_20_03_800_top100_varimp.csv')
features   = features$variable
features   = as.character(features)

####################################################################################
gbmHex <- h2o.gbm(  x=features,
                    y="logSales",
                    training_frame=trainHex,
                    model_id="introGBM",
                    nbins_cats=1115,
                    sample_rate = 0.5,
                    col_sample_rate = 0.5,
                    max_depth = 12,
                    learn_rate=0.5,
                    seed = 12345678,
                    ntrees = 600,
                    validation_frame = validHex)

summary(gbmHex)
(varimps = data.frame(h2o.varimp(gbmHex)))
write.csv(varimps, "./H2O_submits/H2O_GBM_12_05_600_CV_varimp.csv",row.names=F)
h2o.saveModel(gbmHex, path = '/Users/jfdarre/Documents/NYCDS/Project4/H2O_models_GBM_12_05_600_CV', force = FALSE)

cat("Predicting Sales\n")
pred_CV <- as.data.frame(h2o.predict(gbmHex,testHex))
pred_CV <- expm1(pred_CV[,1])

submit_CV <- data.frame(Id=test$Id, Sales=pred_CV)
cat("saving the submission file\n")
write.csv(submit_CV, "./H2O_submits/H2O_GBM_12_05_600_CV",row.names=F)

train_pred   <- as.data.frame(h2o.predict(gbmHex,trainHex))
train_pred   <- expm1(train_pred[,1])
train_pred   <- data.frame(Id=train$Id, Sales=train_pred)
write.csv(train_pred, "./H2O_submits/H2O_GBM_12_05_600_CV_train.csv",row.names=F)

valid_pred   <- as.data.frame(h2o.predict(gbmHex,validHex))
valid_pred   <- expm1(valid_pred[,1])
valid_pred   <- data.frame(Id=valid$Id, Sales=valid_pred)
write.csv(valid_pred, "./H2O_submits/H2O_GBM_12_05_600_CV_valid.csv",row.names=F)


####################################################################################
a = seq(1,100,4)
b = seq(2,100,4)
c = seq(3,100,4)
d = c(2, seq(4,100,4))

####################################################################################
feats_a = features[a]

gbmHex_a <- h2o.gbm(x               = feats_a,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 15,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400,
                    validation_frame = validHex)

summary(gbmHex_a)
h2o.saveModel(gbmHex_a, path = '../H2O_models_GBM_15_05_400_CV_a', force = FALSE)

cat("Predicting Sales\n")
pred_a       <- as.data.frame(h2o.predict(gbmHex_a, testHex))
pred_a       <- expm1(pred_a[,1])
submit_a     <- data.frame(Id = test$Id, Sales = pred_a)
cat("saving the submission file\n")
write.csv(submit_a, "./H2O_submits/H2O_models_GBM_15_05_400_CV_a.csv",row.names=F)

train_pred_a   <- as.data.frame(h2o.predict(gbmHex_a, trainHex))
train_pred_a   <- expm1(train_pred_a[,1])
train_pred_a   <- data.frame(Id = train$Id, Sales = train_pred_a)
write.csv(train_pred_a, "./H2O_submits/H2O_models_GBM_15_05_400_CV_a.csv",row.names=F)

valid_pred_a   <- as.data.frame(h2o.predict(gbmHex_a, validHex))
valid_pred_a   <- expm1(valid_pred_a[,1])
valid_pred_a   <- data.frame(Id = valid$Id, Sales = valid_pred_a)
write.csv(valid_pred_a, "./H2O_submits/H2O_models_GBM_15_05_400_CV_a.csv",row.names=F)

####################################################################################
feats_b = features[b]

gbmHex_b <- h2o.gbm(x               = feats_b,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 15,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400,
                    validation_frame = validHex)

summary(gbmHex_b)
h2o.saveModel(gbmHex_b, path = '../H2O_models_GBM_15_05_400_CV_b', force = FALSE)

cat("Predicting Sales\n")
pred_b       <- as.data.frame(h2o.predict(gbmHex_b, testHex))
pred_b       <- expm1(pred_b[,1])
submit_b     <- data.frame(Id = test$Id, Sales = pred_b)
cat("saving the submission file\n")
write.csv(submit_b, "./H2O_submits/H2O_models_GBM_15_05_400_CV_b.csv",row.names=F)

train_pred_b   <- as.data.frame(h2o.predict(gbmHex_b, trainHex))
train_pred_b   <- expm1(train_pred_b[,1])
train_pred_b   <- data.frame(Id = train$Id, Sales = train_pred_b)
write.csv(train_pred_b, "./H2O_submits/H2O_models_GBM_15_05_400_CV_b.csv",row.names=F)

valid_pred_b   <- as.data.frame(h2o.predict(gbmHex_b, validHex))
valid_pred_b   <- expm1(valid_pred_b[,1])
valid_pred_b   <- data.frame(Id = valid$Id, Sales = valid_pred_b)
write.csv(valid_pred_b, "./H2O_submits/H2O_models_GBM_15_05_400_CV_b.csv",row.names=F)

####################################################################################
feats_c = features[c]

gbmHex_c <- h2o.gbm(x               = feats_c,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 15,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400,
                    validation_frame = validHex)

summary(gbmHex_c)
h2o.saveModel(gbmHex_c, path = '../H2O_models_GBM_15_05_400_CV_c', force = FALSE)

cat("Predicting Sales\n")
pred_c       <- as.data.frame(h2o.predict(gbmHex_c, testHex))
pred_c       <- expm1(pred_c[,1])
submit_c     <- data.frame(Id = test$Id, Sales = pred_c)
cat("saving the submission file\n")
write.csv(submit_c, "./H2O_submits/H2O_models_GBM_15_05_400_CV_c.csv",row.names=F)

train_pred_c   <- as.data.frame(h2o.predict(gbmHex_c, trainHex))
train_pred_c   <- expm1(train_pred_c[,1])
train_pred_c   <- data.frame(Id = train$Id, Sales = train_pred_c)
write.csv(train_pred_c, "./H2O_submits/H2O_models_GBM_15_05_400_CV_c.csv",row.names=F)

valid_pred_c   <- as.data.frame(h2o.predict(gbmHex_c, validHex))
valid_pred_c   <- expm1(valid_pred_c[,1])
valid_pred_c   <- data.frame(Id = valid$Id, Sales = valid_pred_c)
write.csv(valid_pred_c, "./H2O_submits/H2O_models_GBM_15_05_400_CV_c.csv",row.names=F)

####################################################################################
feats_d = features[d]

gbmHex_d <- h2o.gbm(x               = feats_d,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 15,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400,
                    validation_frame = validHex)

summary(gbmHex_d)
h2o.saveModel(gbmHex_d, path = '../H2O_models_GBM_15_05_400_CV_d', force = FALSE)

cat("Predicting Sales\n")
pred_d       <- as.data.frame(h2o.predict(gbmHex_d, testHex))
pred_d       <- expm1(pred_d[,1])
submit_d     <- data.frame(Id = test$Id, Sales = pred_d)
cat("saving the submission file\n")
write.csv(submit_d, "./H2O_submits/H2O_models_GBM_15_05_400_CV_d.csv",row.names=F)

train_pred_d   <- as.data.frame(h2o.predict(gbmHex_d, trainHex))
train_pred_d   <- expm1(train_pred_d[,1])
train_pred_d   <- data.frame(Id = train$Id, Sales = train_pred_d)
write.csv(train_pred_d, "./H2O_submits/H2O_models_GBM_15_05_400_CV_d.csv",row.names=F)

valid_pred_d   <- as.data.frame(h2o.predict(gbmHex_d, validHex))
valid_pred_d   <- expm1(valid_pred_d[,1])
valid_pred_d   <- data.frame(Id = valid$Id, Sales = valid_pred_d)
write.csv(valid_pred_d, "./H2O_submits/H2O_models_GBM_15_05_400_CV_d.csv",row.names=F)

####################################################################################
## Results:

sqrt(sum(rmse(train_pred$Sales, train$Sales))/nrow(train))
sqrt(sum(rmse(train_pred_a$Sales, train$Sales))/nrow(train))
sqrt(sum(rmse(train_pred_b$Sales, train$Sales))/nrow(train))
sqrt(sum(rmse(train_pred_c$Sales, train$Sales))/nrow(train))
sqrt(sum(rmse(train_pred_d$Sales, train$Sales))/nrow(train))

sqrt(sum(rmse(valid_pred$Sales, valid$Sales))/nrow(valid))
sqrt(sum(rmse(valid_pred_a$Sales, valid$Sales))/nrow(valid))
sqrt(sum(rmse(valid_pred_b$Sales, valid$Sales))/nrow(valid))
sqrt(sum(rmse(valid_pred_c$Sales, valid$Sales))/nrow(valid))
sqrt(sum(rmse(valid_pred_d$Sales, valid$Sales))/nrow(valid))


####################################################################################
grid_search = data.frame()
max_weight = 4
for (i in 0:max_weight) {
  for (j in 0:(max_weight-i)) {
    for (k in 0:(max_weight-i-j)) {
      for (l in 0:(max_weight-i-j-k)) {
        pred = ((i-max_weight/2)*valid_pred_a$Sales +
                (j-max_weight/2)*valid_pred_b$Sales +
                (k-max_weight/2)*valid_pred_c$Sales +
                (l-max_weight/2)*valid_pred_d$Sales +
                (3*max_weight-i-j-k-l)*valid_pred$Sales)/max_weight
        temp = data.frame(   a=(i-max_weight/2),
                             b=(j-max_weight/2),
                             c=(k-max_weight/2), 
                             d=(j-max_weight/2), 
                          main=3*max_weight-i-j-k-l,
                          rmse=sqrt(sum(rmse(pred, valid$Sales))/nrow(valid)))
        grid_search = rbind(grid_search, temp)
      }
    }
  }
}


filter(grid_search, rmse == min(grid_search[,6]))
head(grid_search[with(grid_search, order(rmse)),],20)
grid_search

####################################################################################
grid_search = data.frame()
max_weight = 6
weight_range = -max_weight:max_weight
for (i in weight_range) {
  for (j in weight_range) {
    for (k in weight_range) {
      for (l in weight_range) {
        for (m in weight_range) {
          if (i+j+k+l+m == 1) {
            pred = ((i)*valid_pred_a$Sales +
                    (j)*valid_pred_b$Sales +
                    (k)*valid_pred_c$Sales +
                    (l)*valid_pred_d$Sales +
                    (m)*valid_pred$Sales)
            temp = data.frame(   a=(i),
                                 b=(j),
                                 c=(k), 
                                 d=(j), 
                              main=(m),
                              rmse=sqrt(sum(rmse(pred, valid$Sales))/nrow(valid)))
            grid_search = rbind(grid_search, temp)
          }
        }
      }
    }
  }
}


filter(grid_search, rmse == min(grid_search[,6]))
head(grid_search[with(grid_search, order(rmse)),],200)
grid_search

####################################################################################
mean(train_pred$Sales)
mean(train_pred_a$Sales)
mean(train_pred_b$Sales)
mean(train_pred_c$Sales)
mean(train_pred_d$Sales)

pred_ensemb  <- (20*pred2 - 4*pred_a - 2*pred_b - 2*pred_c - 2*pred_d)/10
mean(pred_ensemb)
submit_d     <- data.frame(Id = test$Id, Sales = pred_ensemb)
cat("saving the submission file\n")
write.csv(submit_d, "./H2O_submits/h2o_GBM_ensemble_v1.csv",row.names=F)


####################################################################################

grid_search = data.frame()
max_weight = 4
weight_range = -max_weight:max_weight
for (i in weight_range) {
  for (j in weight_range) {
    for (k in weight_range) {
      for (l in weight_range) {
        for (m in weight_range) {
          if (i+j+k+l+m == 1) {
            temp = data.frame(   a=i,
                                 b=j,
                                 c=k, 
                                 d=l, 
                              main=m,
                              rmse=1)
            grid_search = rbind(grid_search, temp)
          }
        }
      }
    }
  }
}

grid_search
sum(grid_search$a)
sum(grid_search$b)
sum(grid_search$c)
sum(grid_search$d)
sum(grid_search$main)
sum(grid_search$rmse)

filter(grid_search, rmse == min(grid_search[,6]))
head(grid_search[with(grid_search, order(rmse)),],20)



