# H2O GBM script version 1
library(caret)
library(plyr)
library(dplyr)
library(data.table)  
library(h2o)

cat("reading the train and test data (with data.table) \n")
train <- fread("../data/train3.csv",stringsAsFactors = T)
test  <- fread("../data/test3.csv",stringsAsFactors = T)
store <- fread("./input/store.csv",stringsAsFactors = T)
train <- train[Sales > 0,]  ## We are not judged on 0 sales records in test set


## create stratified folds for cross-validation
# folds <- createFolds(factor(store$Store), k = 10, list = FALSE)
# store$fold <- folds
# ddply(store, 'fold', summarise, prop=mean(store$fold)/10)

train <- merge(train,store,by="Store")
test <- merge(test,store,by="Store")

# cat("train data column names and details\n")
# summary(train)
# cat("test data column names and details\n")
# summary(test)

## more care should be taken to ensure the dates of test can be projected from train
## decision trees do not project well, so you will want to have some strategy here, if using the dates
train[,Date:=as.Date(Date)]
test[,Date:=as.Date(Date)]

# competition feature
train$Competition <- (sqrt(max(train$CompetitionDistance, na.rm = TRUE)-train$CompetitionDistance))*
  (((train$year - train$CompetitionOpenSinceYear) * 12) - (train$CompetitionOpenSinceMonth-train$month))

test$Competition <- (sqrt(max(test$CompetitionDistance, na.rm = TRUE)-test$CompetitionDistance))*
  (((test$year - test$CompetitionOpenSinceYear) * 12) - (test$CompetitionOpenSinceMonth-test$month))

str(train)
str(test)

# Set appropriate variables to factors
for (j in c("Store", "DayOfWeek", "Promo", 
            "year", "month", "day", "PromoFirstDate",
#            "day", "day_of_year", "week_of_year", "PromoFirstDate",
            "State", "PromoSecondDate", #"DayBeforeClosed", "DayAfterClosed",
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2", "Promo2SinceWeek", "Promo2SinceYear")) {
  train[[j]] <- as.factor(train[[j]])
  test[[j]]  <- as.factor(test[[j]])
}

## log transformation to not be as sensitive to high sales
## decent rule of thumb: 
##     if the data spans an order of magnitude, consider a log transform
train[,logSales:=log1p(Sales)]
str(train)
str(test)

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

# create validation and training set
# trainHex<-as.h2o(train[year <15 | month <6,],destination_frame = "trainHex")
# validHex<-as.h2o(train[year == 15 & month >= 6,],destination_frame = "validHex")
# dim(trainHex); dim(validHex)

## Load data into cluster from R
trainHex<-as.h2o(train)
features<-names(train)[!(names(train) %in% c("Id","Date","Sales","logSales", "Customers"))]
features

####################################################################################
gbmHex <- h2o.gbm(x=features,
                  y="logSales",
                  training_frame=trainHex,
                  model_id="introGBM",
                  nbins_cats=1115, #5, 12, 223, 1115,
                  sample_rate = 0.5,
                  col_sample_rate = 0.5,
                  max_depth = 10,
                  learn_rate=0.05,
                  seed = 12345678,
                  ntrees = 20
                  )

summary(gbmHex)
(varimps = data.frame(h2o.varimp(gbmHex)))
sumup(model = gbmHex, trainHex = trainHex, train = train)
gbm_10_05_20_forVarImp_v2 = gbmHex
h2o.saveModel(gbm_10_05_20_forVarImp_v2, path = '../H2O_models_GBM_varImp_v2', force = FALSE)

####################################################################################
features2 = varimps$variable[1:100]

gbmHex2 <- h2o.gbm( x=features2,
                    y="logSales",
                    training_frame=trainHex,
                    model_id="introGBM",
                    nbins_cats=1115,
                    sample_rate = 0.5,
                    col_sample_rate = 0.5,
                    max_depth = 20,
                    learn_rate=0.03,
                    seed = 12345678,
                    ntrees = 800)

summary(gbmHex2)
(varimps2 = data.frame(h2o.varimp(gbmHex2)))
sumup(model = gbmHex2, trainHex = trainHex, train = train)
gbm_20_03_800 = gbmHex2
h2o.saveModel(gbm_20_03_800, path = '/Users/jfdarre/Documents/NYCDS/Project4/H2O_models_GBM_20_03_800', force = FALSE)

cat("Predicting Sales\n")
testHex2     <- as.h2o(test)
predictions2 <- as.data.frame(h2o.predict(gbmHex2,testHex2))
pred2        <- expm1(predictions2[,1])
summary(pred2)

submission2  <- data.frame(Id=test$Id, Sales=pred2)
cat("saving the submission file\n")
write.csv(submission2, "./H2O_submits/h2o_GBM_20_03_800_top100.csv",row.names=F)

train_pred   <- as.data.frame(h2o.predict(gbmHex2,trainHex))
train_pred   <- expm1(train_pred[,1])
train_pred   <- data.frame(Id=train$Id, Sales=train_pred)
write.csv(train_pred, "./H2O_submits/h2o_GBM_20_03_800_top100_train.csv",row.names=F)
write.csv(varimps2, "./H2O_submits/h2o_GBM_20_03_800_top100_varimp.csv",row.names=F)

####################################################################################
a = c(1:2, seq(3,100,4))
b = c(1:2, seq(4,100,4))
c = c(1:2, seq(5,100,4))
d = c(1:2, seq(6,100,4))

####################################################################################
feats_a = c(varimps2$variable[a])

gbmHex_a <- h2o.gbm(x               = feats_a,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 20,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400)

summary(gbmHex_a)
gbm_15_05_300_a = gbmHex_a
h2o.saveModel(gbm_15_05_300_a, path = '../H2O_models_GBM_15_05_300_v4_a', force = FALSE)

cat("Predicting Sales\n")
testHex      <- as.h2o(test)
pred_a       <- as.data.frame(h2o.predict(gbmHex_a, testHex))
pred_a       <- expm1(pred_a[,1])
submit_a     <- data.frame(Id = test$Id, Sales = pred_a)
cat("saving the submission file\n")
write.csv(submit_a, "./H2O_submits/h2o_GBM_20_05_300_v3_a.csv",row.names=F)

train_pred_a   <- as.data.frame(h2o.predict(gbmHex_a, trainHex))
train_pred_a   <- expm1(train_pred_a[,1])
train_pred_a   <- data.frame(Id = train$Id, Sales = train_pred_a)
write.csv(train_pred_a, "./H2O_submits/h2o_GBM_20_05_300_v3_a_train.csv",row.names=F)

####################################################################################
feats_b = c(varimps2$variable[b])

gbmHex_b <- h2o.gbm(x               = feats_b,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 20,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400)

summary(gbmHex_b)
gbm_15_05_300_b = gbmHex_b
h2o.saveModel(gbm_15_05_300_b, path = '../H2O_models_GBM_15_05_300_v4_b', force = FALSE)

cat("Predicting Sales\n")
testHex      <- as.h2o(test)
pred_b       <- as.data.frame(h2o.predict(gbmHex_b, testHex))
pred_b       <- expm1(pred_b[,1])
submit_b     <- data.frame(Id = test$Id, Sales = pred_b)
cat("saving the submission file\n")
write.csv(submit_b, "./H2O_submits/h2o_GBM_20_05_300_v3_b.csv",row.names=F)

train_pred_b   <- as.data.frame(h2o.predict(gbmHex_b, trainHex))
train_pred_b   <- expm1(train_pred_b[,1])
train_pred_b   <- data.frame(Id = train$Id, Sales = train_pred_b)
write.csv(train_pred_b, "./H2O_submits/h2o_GBM_20_05_300_v3_b_train.csv",row.names=F)

####################################################################################
feats_c = c(varimps2$variable[c])

gbmHex_c <- h2o.gbm(x               = feats_c,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 20,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400)

summary(gbmHex_c)
gbm_15_05_300_c = gbmHex_c
h2o.saveModel(gbm_15_05_300_c, path = '../H2O_models_GBM_15_05_300_v4_c', force = FALSE)

cat("Predicting Sales\n")
testHex      <- as.h2o(test)
pred_c       <- as.data.frame(h2o.predict(gbmHex_c, testHex))
pred_c       <- expm1(pred_c[,1])
submit_c     <- data.frame(Id = test$Id, Sales = pred_c)
cat("saving the submission file\n")
write.csv(submit_c, "./H2O_submits/h2o_GBM_20_05_300_v3_c.csv",row.names=F)

train_pred_c   <- as.data.frame(h2o.predict(gbmHex_c, trainHex))
train_pred_c   <- expm1(train_pred_c[,1])
train_pred_c   <- data.frame(Id = train$Id, Sales = train_pred_c)
write.csv(train_pred_c, "./H2O_submits/h2o_GBM_20_05_300_v3_c_train.csv",row.names=F)

####################################################################################
feats_d = c(varimps2$variable[d])

gbmHex_d <- h2o.gbm(x               = feats_d,
                    y               = "logSales",
                    training_frame  = trainHex,
                    model_id        = "introGBM",
                    nbins_cats      = 1115,
                    sample_rate     = 0.5,
                    col_sample_rate = 0.5,
                    max_depth       = 20,
                    learn_rate      = 0.05,
                    seed            = 12345678,
                    ntrees          = 400)

summary(gbmHex_d)
gbm_15_05_300_d = gbmHex_d
h2o.saveModel(gbm_15_05_300_d, path = '../H2O_models_GBM_15_05_300_v4_d', force = FALSE)

cat("Predicting Sales\n")
testHex      <- as.h2o(test)
pred_d       <- as.data.frame(h2o.predict(gbmHex_d, testHex))
pred_d       <- expm1(pred_d[,1])
submit_d     <- data.frame(Id = test$Id, Sales = pred_d)
cat("saving the submission file\n")
write.csv(submit_d, "./H2O_submits/h2o_GBM_20_05_300_v3_d.csv",row.names=F)

train_pred_d   <- as.data.frame(h2o.predict(gbmHex_d, trainHex))
train_pred_d   <- expm1(train_pred_d[,1])
train_pred_d   <- data.frame(Id = train$Id, Sales = train_pred_d)
write.csv(train_pred_d, "./H2O_submits/h2o_GBM_20_05_300_v3_d_train.csv",row.names=F)

####################################################################################
grid_search = data.frame()
max_weight = 10
for (i in 0:max_weight) {
  for (j in 0:(max_weight-i)) {
    for (k in 0:(max_weight-i-j)) {
      for (l in 0:(max_weight-i-j-k)) {
        pred = ((i-max_weight/2)*train_pred_a$Sales +
                (j-max_weight/2)*train_pred_b$Sales +
                (k-max_weight/2)*train_pred_c$Sales +
                (l-max_weight/2)*train_pred_d$Sales +
                (3*max_weight-i-j-k-l)*train_pred$Sales)/max_weight
        temp = data.frame(   a=(i-max_weight/2),
                             b=(j-max_weight/2),
                             c=(k-max_weight/2), 
                             d=(j-max_weight/2), 
                          main=3*max_weight-i-j-k-l,
                          rmse=sqrt(sum(rmse(pred, train$Sales))/nrow(train)))
        grid_search = rbind(grid_search, temp)
      }
    }
  }
}

grid_search
filter(grid_search, rmse == min(grid_search[,6]))
head(grid_search[with(grid_search, order(rmse)),],20)

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
features3 = varimps$variable[1:80]

gbmHex3 <- h2o.gbm( x=features3,
                    y="logSales",
                    training_frame=trainHex,
                    model_id="introGBM",
                    nbins_cats=1115,
                    sample_rate = 0.5,
                    col_sample_rate = 0.5,
                    max_depth = 12,
                    learn_rate=0.05,
                    seed = 12345678,
                    ntrees = 800)

summary(gbmHex3)
(varimps3 = data.frame(h2o.varimp(gbmHex3)))
sumup(model = gbmHex3, trainHex = trainHex, train = train)
gbm_12_05_800_top80 = gbmHex3
h2o.saveModel(gbm_12_05_800_top80, 
              path = '/Users/jfdarre/Documents/NYCDS/Project4/H2O_models_GBM_12_05_800_top80', force = FALSE)

cat("Predicting Sales\n")
testHex3     <- as.h2o(test)
predictions3 <- as.data.frame(h2o.predict(gbmHex3,testHex3))
pred3        <- expm1(predictions3[,1])
summary(pred3)

submission3  <- data.frame(Id=test$Id, Sales=pred3)
cat("saving the submission file\n")
write.csv(submission3, "./H2O_submits/h2o_GBM_12_05_800_top80_v2.csv",row.names=F)

train_pred3   <- as.data.frame(h2o.predict(gbmHex3,trainHex))
train_pred3   <- expm1(train_pred3[,1])
train_pred3   <- data.frame(Id=train$Id, Sales=train_pred3)
write.csv(train_pred3, "./H2O_submits/h2o_GBM_12_05_800_top80_train_v2.csv",row.names=F)
write.csv(varimps3, "./H2O_submits/h2o_GBM_12_05_800_top80_varimp_2.csv",row.names=F)

####################################################################################

pred_ensemb2   <- (2*pred2 + pred_a + pred_b + pred_c + pred_d + 6*pred3)/12
mean(pred_ensemb2)
submit_ensemb2 <- data.frame(Id = test$Id, Sales = pred_ensemb2)
cat("saving the submission file\n")
write.csv(submit_d, "./H2O_submits/h2o_GBM_ensemble_v2.csv",row.names=F)




####################################################################################

sub1 <- fread("./H2O_submits/h2o_30_60.csv",stringsAsFactors = T)
sub2 <- fread("./H2O_submits/h2o_30_80.csv",stringsAsFactors = T)
sub3 <- fread("./H2O_submits/h2o_50_65.csv",stringsAsFactors = T)
sub4 <- fread("./H2O_submits/h2o_GBM_20_05_300.csv",stringsAsFactors = T)
sub5 <- fread("./H2O_submits/h2o_GBM_20_05_300_v3.csv",stringsAsFactors = T)

mean(sub1$Sales)
mean(sub2$Sales)
mean(sub3$Sales)
mean(sub4$Sales)
mean(sub5$Sales)

new_sub = (sub1+sub2+2*sub4+2*sub5)/6
mean(new_sub$Sales)
write.csv(new_sub, "./H2O_submits/ensemble_test_v1.csv",row.names=F)

new_sub2 = (sub1+sub2+sub4+sub5)/4
mean(new_sub2$Sales)
write.csv(new_sub2, "./H2O_submits/ensemble_test_v2.csv",row.names=F)

####################################################################################
