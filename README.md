# Overview

As part of a Kaggle competition, we were challenged by Rossmann, the second largest chain of German drug stores, to predict the daily sales for 6 weeks into the future for more than 1,000 stores. Exploratory data analysis revealed several novel features, including spikes in sales prior to, and preceding store refurbishment. We also engineered several novel features by the inclusion of external data including Google Trends, macroeconomic data, as well as weather data. We then used H20, a fast, scalable parallel-processing engine for machine learning, to build predictive models utilizing random forests, gradient boosting machines, as well as deep learning. Lastly, we combined these models using different ensemble methods to obtain better predictive performance.

Training data was provided for 1,115 Rossmann stores from January 1st 2013 through July 31st 2015 .The task was to forecast 6 weeks (August 1st 2015 through September 17th 2015) of sales for 856 of the Rossmann stores identified within the testing data.

## Data Sets

*   TRAIN.CSV - historical data including sales
*   TEST.CSV - historical data excluding sales
*   SAMPLE_SUBMISSION.CSV - sample submission file in the correct format
*   STORE.CSV - supplemental information describing each of the stores

## Data Fields

*   Id - represents a (Store, Date) duple within the test set
*   Store - Unique Id for each store
*   Sales - The turnover for any given day (variable to be predicted)
*   Customers - The number of customers on a given day
*   Open - An indicator for whether the store was open: 0 = closed, 1 = open
*   StateHoliday - Indicates a state holiday
    *   Normally all stores, with few exceptions, are closed on state holidays.
    *   All schools are closed on public holidays and weekends.
    *   a = public holiday
    *   b = Easter holiday
    *   c = Christmas
    *   0 = None
*   SchoolHoliday - Indicates if the (Store, Date) was affected by the closure of public schools
*   StoreType - Differentiates between 4 different store models:
    *   a, b, c, d
*   Assortment - Describes an assortment level:
    *   a = basic, b = extra, c = extended
*   CompetitionDistance - Distance in meters to the nearest competitor store
*   CompetitionOpenSince [Month/Year] - Approximate year and month of the time the nearest competitor was opened
*   Promo - Indicates whether a store is running a promo on that day
*   Promo2 - Continuing and consecutive promotion for some stores:
    *   0 = store is not participating
    *   1 = store is participating
*   Promo2Since [Year/Week] - Describes the year and calendar week when the store started participating in Promo2
*   PromoInterval - Describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew.
    *   “Feb,May,Aug,Nov” means each round starts in February, May, August, November of any given year for that store

## Exploratory Data Analysis

Exploratory data analysis was performed in ipython notebook and R. Findings discovered throughout the EDA process were all addressed during data cleaning and feature engineering. Courtesy of fellow Kaggler Paul Shearer for creating a fantastic dygraph to display all data.

[![Picture1](http://2igww43ul7xe3nor5h2ah1yq.wpengine.netdna-cdn.com/wp-content/uploads/2015/12/Picture1.png)](http://2igww43ul7xe3nor5h2ah1yq.wpengine.netdna-cdn.com/wp-content/uploads/2015/12/Picture1.png)

## Data Cleaning

1.  Impute Open = 1 for missing Open test dataset
    *   Special case found during EDA. Store 622 has several missing dates which are all weekdays with sales recorded.
2.  Set Open = 0 when Sales = 0 OR Customers = 0
3.  Standardize StateHoliday due to the use of character 0 and integer 0
4.  Separate date column into year, month and day. Also convert Date column to type ‘date’ and extract:
    *   day_of_year
    *   week_of_year
    *   quarter
    *   month_start
    *   month_end
    *   quarter_start
    *   quarter_end
5.  Remove observations where stores are closed. These values can be hard coded after prediction since no sales can occur with a closed store.
6.  Set store as factor.
7.  Merge store dataset with train and test
8.  Stores.csv contained an abundance of missing values. Machine learning methods were chosen with this in mind. The following methods that were implemented handle missing values with ease. Their methods are described below.
    *   Distributed Random Forest and Gradient Boosting Machine treat missing (NA) factor levels as the smallest value present (left-most in the bins), which can go left or right for any split, and unseen factor levels (the case here) to always go left in any split.
    *   Deep Learning by default makes an extra input neuron for missing and unseen categorical levels which can remain untrained if there were no such instances in the training data, leading to some random contribution to the next layer during testing.
9.  Experimenting with variables as factors:
    *   We experimented with setting features to factors in order to see their effects upon the MSE and residual errors. We should note that H20 can deal with large numbers of factors and the categorical data does not need to be one-hot encoded. H2O does not expand categorical data into dummy variables, but instead uses a bitset to determine which categorical levels go left or right on each split.
    *   H2O is also more accurate than R/Python. I think the reason for that is dealing properly with the categorical variables, i.e. internally in the algo rather than working from a previously 1-hot encoded dataset where the link between the dummies belonging to the same original variable is lost. This is by the way how the R package should work if the number of categories is small (but not in our case).
        *   Train a model on data that has a categorical predictor (column) with levels B,C,D (and no other levels). Let’s call these levels the “training set domain”: {B,C,D}
        *   During scoring, a test set has only rows with levels A,C,E for that column, the “test set domain”: {A,C,E}
        *   For scoring, we construct the joint “scoring domain”: {B,C,D,A,E}, which is the training domain with the extra test set domain entries appended.
        *   Each model can handle these extra levels {A,E} during scoring separately.The way H2O deals with categories is not only more proper and gets better AUC, but it is makes it faster and more memory efficient. See [Categorical variables with random forest](https://groups.google.com/forum/#!topic/h2ostream/7dOJ5X8KCT4) for more information.In addition, most machine learning tools break when you try to predict with a new level for a categorical input that was not present in the training set. However, H2O is able to handle such a situation. Here’s an example of how this works:
    *   <span style="line-height: 1.5;">See</span> [prediction with categorical variable with a new level](https://groups.google.com/forum/m/#!msg/h2ostream/f1elOH4Pbsc/lNCXQyCQBwAJ) <span style="line-height: 1.5;">for more information.</span>
10.  We use a log transformation for the sales in order not to be as sensitive to high sales. A decent rule of thumb is if the data spans an order of magnitude, then consider using a log transform.

<div id="data-cleaning-and-munching-steps-below" class="section level2">

## Feature Engineering

1.  Promotion First Day and Promotion Second Day
2.  DayBeforeClosed
3.  DayAfterClosed
4.  Store Open on Sunday
5.  Closed
6.  Refurbishment period
    *   EDA lead to a discovering a spike in sales before and after a period of extended close indicating a clearance and grand opening sale
    *   One thing to note was that Some stores in the dataset were temporarily closed for refurbishment.
7.  DaysBeforeRefurb
8.  DaysAfterRefurb
9.  Competition Feature
    *   We create this new feature by taking the square root of the difference between the maximum distance of a competitor store among all the stores and the distance of a competitor store for an individual store, times the time since a competitor opened:

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;"># Competition Feature</span>
<span style="color: #888888;">train$Competition <-</span> 
 <span style="color: #888888;">(sqrt(max(train$CompetitionDistance, na.rm = TRUE) -</span> 
 <span style="color: #888888;">train$CompetitionDistance)) *</span>
 <span style="color: #888888;">(((train$year - train$CompetitionOpenSinceYear) * 12) -</span> 
 <span style="color: #888888;">(train$CompetitionOpenSinceMonth-train$month))</span>

<span style="color: #888888;">test$Competition <-</span> 
 <span style="color: #888888;">(sqrt(max(test$CompetitionDistance, na.rm = TRUE) -</span> 
 <span style="color: #888888;">test$CompetitionDistance))*</span>
 <span style="color: #888888;">(((test$year - test$CompetitionOpenSinceYear) * 12) -</span> 
 <span style="color: #888888;">(test$CompetitionOpenSinceMonth-test$month))</span></pre>

</div>

## Open Data Sources

</div>

<div id="data-cleaning-and-munching-steps-below" class="section level2">

*   German States derived from StateHoliday
*   German State Weather
*   Google Trends
*   Export into CSV

# Introduction to H20

H2O is an open source math & machine learning engine for big data that brings distribution and parallelism to powerful algorithms while keeping the widely used languages such as R, Spark, Python, Java, and JSON as an API. Using in-memory compression, H2O handles billions of data rows in-memory, even with a small cluster. H2O includes many common machine learning algorithms, such as generalized linear modeling (linear regression, logistic regression, etc.), Na ̈ıve Bayes, principal components analysis, k-means clustering, and others. H2O also implements best-in-class algorithms at scale, such as distributed random forest, gradient boosting, and deep learning.

</div>

<div id="h2o-implementation" class="section level2">

## H2O Process

*   Have correct version of Java installed
*   Start H20 instance
*   Set features to factors (and test)
*   Create validation set
*   Tune Parameters for each model (manual or grid search)
*   Use model to make predictions on test set
*   Iterate

</div>

<div id="load-libraries-into-r" class="section level2">

## Load Libraries into R

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;">library(caret)</span>
<span style="color: #888888;">library(data.table)</span> 
<span style="color: #888888;">library(h2o)</span>
<span style="color: #888888;">library(plyr)</span></pre>

</div>

</div>

<div id="start-h20-cluster-with-all-available-threads" class="section level2">

## Initialize H2O Cluster With All Available Threads

One should use `h2o.shutdown()` if changing parameters below. Also, setting `assertion = FALSE` seems to help with stability of H20.

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;">h2o.init(nthreads=-1,max_mem_size='8G', assertion = FALSE)</span>
<span style="color: #888888;">## Successfully connected to http://127.0.0.1:54321/</span> 
<span style="color: #888888;">##</span> 
<span style="color: #888888;">## R is connected to the H2O cluster:</span> 
<span style="color: #888888;">##     H2O cluster uptime:         16 hours 46 minutes</span> 
<span style="color: #888888;">##     H2O cluster version:        3.6.0.8</span> 
<span style="color: #888888;">##     H2O cluster name:           H2O_started_from_R_2015_ukz280</span> 
<span style="color: #888888;">##     H2O cluster total nodes:    1</span> 
<span style="color: #888888;">##     H2O cluster total memory:   7.98 GB</span> 
<span style="color: #888888;">##     H2O cluster total cores:    8</span> 
<span style="color: #888888;">##     H2O cluster allowed cores:  8</span> 
<span style="color: #888888;">##     H2O cluster healthy:        TRUE</span>
<span style="color: #888888;">## IP Address: 127.0.0.1</span> 
<span style="color: #888888;">## Port      : 54321</span> 
<span style="color: #888888;">## Session ID: _sid_ac1406fd65438164da7936d76cfe44b2</span> 
<span style="color: #888888;">## Key Count : 0</span></pre>

</div>

<div id="reading-the-train-and-test-data-with-data.table" class="section level2">

## Read Test and Train Data

Data.table was used as a means of reading in data due to its increased efficiency for data frame manipulation over dplyr.

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;">train <- fread("KaggleProject/data/train_states_R_v8.csv",</span>
 <span style="color: #888888;">stringsAsFactors = T)</span>
<span style="color: #888888;">test  <- fread("KaggleProject/data/test_states_R_v8.csv",</span>
 <span style="color: #888888;">stringsAsFactors = T, showProgress=FALSE)
store <- fread("input/store.csv",
                stringsAsFactors = T)</span></pre>

</div>

</div>

## Create Stratified Folds For Cross-Validation

<div id="create-stratified-folds-for-cross-validation." class="section level2">

The Rossmann dataset is a “pooled-repeated measures” dataset, whereby multiple observations from different stores are grouped together. Hence, the internal cross-validation has to be done in an “honest” manner, i.e., all the observations from one store must belong to a single fold. Otherwise, it can lead to overfitting. Creating stratified folds for cross-validation can be easily achieved by utilizing the `createFolds` method from the Caret package in R. Since the stores dataset is a list of each store with one store per row, we can create the folds in the stores dataset prior to merging this dataset with the train and test datasets.

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;">folds <- createFolds(factor(store$Store), k = 10, list = FALSE)</span>
<span style="color: #888888;">store$fold <- folds</span>
<span style="color: #888888;">ddply(store, 'fold', summarise, prop=mean(store$fold)/10)</span>
<span style="color: #888888;">##    fold      prop</span>
<span style="color: #888888;">## 1     1 0.5598206</span>
<span style="color: #888888;">## 2     2 0.5598206</span>
<span style="color: #888888;">## 3     3 0.5598206</span>
<span style="color: #888888;">## 4     4 0.5598206</span>
<span style="color: #888888;">## 5     5 0.5598206</span>
<span style="color: #888888;">## 6     6 0.5598206</span>
<span style="color: #888888;">## 7     7 0.5598206</span>
<span style="color: #888888;">## 8     8 0.5598206</span>
<span style="color: #888888;">## 9     9 0.5598206</span>
<span style="color: #888888;">## 10   10 0.5598206</span></pre>

</div>

</div>

<div id="set-appropriate-variables-to-factors" class="section level2">

## H2O Training and Validation Test Sets

</div>

<div id="create-validation-and-training-set-and-load-the-data-frames-into-h20." class="section level1">

We simply split the training set by date; the training set is simply all rows prior to June 2015 and the validation set are rows June 2015 and on. We then check the dimensions of the training and test sets.

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;">trainHex<-as.h2o(train[year <2015 | month <6,],
          destination_frame = "trainHex")</span>
<span style="color: #888888;">validHex<-as.h2o(train[year == 2015 & month >= 6,],
          destination_frame = "validHex")</span> 
<span style="color: #888888;">dim(trainHex)
dim(validHex)</span> 
<span style="color: #888888;">## [1] 785727     37</span>
<span style="color: #888888;">## [1] 58611    37</span></pre>

</div>

</div>

<div id="specify-the-feature-set-to-use" class="section level1">

## Feature Set For Training

We exclude the Store `Id`, `Date`, `Sales`, `LogSales` and `Customers`. The Store `Id` is only used as a identifier; we split the date into different components (day, month, and year); the Sales and Log of the Sales are what we predicting; and the we are not given the customers in the test and, hence, cannot use it as a feature in the training set.

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;">features<-names(train)[!(names(train) %in%</span> 
 <span style="color: #888888;">c("Id","Date","Sales","logSales",</span> 
 <span style="color: #888888;">"Customers", "Closed", "fold"))]</span>
<span style="color: #888888;">features</span>
<span style="color: #888888;">##  [1] "Store"                     "DayOfWeek"</span> 
<span style="color: #888888;">##  [3] "Open"                      "Promo"</span> 
<span style="color: #888888;">##  [5] "StateHoliday"              "SchoolHoliday"</span> 
<span style="color: #888888;">##  [7] "year"                      "month"</span> 
<span style="color: #888888;">##  [9] "day"                       "day_of_year"</span> 
<span style="color: #888888;">## [11] "week_of_year"              "PromoFirstDate"</span> 
<span style="color: #888888;">## [13] "PromoSecondDate"           "DayBeforeClosed"</span> 
<span style="color: #888888;">## [15] "DayAfterClosed"            "SundayStore"</span> 
<span style="color: #888888;">## [17] "DayBeforeRefurb"           "DayAfterRefurb"</span> 
<span style="color: #888888;">## [19] "DaysBeforeRefurb"          "DaysAfterRefurb"</span> 
<span style="color: #888888;">## [21] "State"                     "StoreType"</span> 
<span style="color: #888888;">## [23] "Assortment"                "CompetitionDistance"</span> 
<span style="color: #888888;">## [25] "CompetitionOpenSinceMonth" "CompetitionOpenSinceYear"</span> 
<span style="color: #888888;">## [27] "Promo2"                    "Promo2SinceWeek"</span> 
<span style="color: #888888;">## [29] "Promo2SinceYear"           "PromoInterval"</span> 
<span style="color: #888888;">## [31] "Competition"</span></pre>

</div>

# Random Forest

#### Intuition

*   Average an ensemble of weakly predicting (larger) trees where each tree is de-correlated from all other trees
*   Bootstrap aggregation (bagging)
*   Fits many trees against different samples of the data and average together

#### Conceptual

*   Combine multiple decision trees, each fit to a random sample of the original data
*   Random samples
*   Rows / Columns
*   Reduce variance wtih minimal increase in bias

#### Strengths

*   Ease of use with limited well-established default parameters
*   Robust
*   Competitive accuracy for most data sets
*   Random forest combines trees and hence incorporates most of the advantages of trees like handling missing values in variable, suiting for both classification and regression, handling highly non-linear interactions and classification boundaries.
*   In addition, Random Forest gives built-in estimates of accuracy, gives automatic variable selection. variable importance, handles wide data – data with more predictors than observations and works well off the shelf – needs no tuning, can get results very quickly. * The runtimes are quite fast, and they are able to deal with unbalanced and missing data.

#### Weaknesses

*   Slow to score
*   Lack of transparency
*   When used for regression they cannot predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy. However, the best test of any algorithm is determined by how well it works upon a particular data set.

## Train a Random Forest Using H20

We should note that we used the stratified folds created in the step above, but H20 also has internal cross-validation (by setting `nfolds` to the desired number of cross-validation folds). [![RF](http://2igww43ul7xe3nor5h2ah1yq.wpengine.netdna-cdn.com/wp-content/uploads/2015/12/RF.jpg)](http://2igww43ul7xe3nor5h2ah1yq.wpengine.netdna-cdn.com/wp-content/uploads/2015/12/RF.jpg) A random forest is an ensemble of decision trees that will output a prediction value. An ensemble model combines the results from different models. A Random Forest is combination of classification and regression. The result from an ensemble model is usually better than the result from one of the individual models. In Random Forest, each decision tree is constructed by using a random subset of the training data that has predictors with known response. After you have trained your forest, you can then pass each test row through it, in order to output a prediction. The goal is to predict the response when it’s unknown. The response can be categorical(classification) or continuous (regression). In a decision tree, an input is entered at the top and as it traverses down the tree the data gets bucketed into smaller and smaller sets. The random forest takes the notion of decision trees to the next level by combining trees. Thus, in ensemble terms, the trees are weak learners and the random forest is a strong learner.

</div>

</div>

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;">rfHex <- h2o.randomForest(x=features,</span>
 <span style="color: #888888;">y="logSales",</span> 
 <span style="color: #888888;">model_id="introRF",</span>
 <span style="color: #888888;">training_frame=trainHex,</span>
 <span style="color: #888888;">validation_frame=validHex,</span>
 <span style="color: #888888;">mtries = -1, # default</span>
 <span style="color: #888888;">sample_rate = 0.632, # default</span>
 <span style="color: #888888;">ntrees = 100,</span>
 <span style="color: #888888;">max_depth = 30,</span>
 <span style="color: #888888;">nbins_cats = 1115, ## allow it to fit store ID</span>
 <span style="color: #888888;">nfolds = 0,</span>
 <span style="color: #888888;">fold_column="fold",</span>
 <span style="color: #888888;">seed = 12345678 #Seed for random numbers (affects sampling)</span>
<span style="color: #888888;">)</span></pre>

</div>

<div id="start-h20-cluster-with-all-available-threads" class="section level2">

<div id="specify-the-feature-set-to-use" class="section level1">

<div id="key-parameters-for-random-forests" class="section level2">

## Key Parameters for Random Forests

The key parameters for the Random Forest model on an H2O frame include:

*   `x`: A vector containing the names or indices of the predictor variables to use in building the GBM model. We have defined `x` to be the features to consider in building our model.
*   `y`: The name or index of the response variable. In our case, the log of the Sales.
*   `training_frame`: An H2O Frame object containing the variables in the model. In our case, this was the subset of the `train` dataset defined above.
*   `model_id`: The unique id assigned to the resulting model.
*   `validation_frame`: An H2O Frame object containing the variables in the model. In our case, this was the subset of the train dataset defined above.
*   `mtries`: At each iteration, a randomly chosen subset of the features in the training data is selected and evaluated to define the optimal split of that subset. `Mtries` specifies the number of features to be selected from the whole set. If set to -1, defaults to p/3 for regression, where p is the number of predictors.
*   `sample_rate`: The sampling rate at each split.
*   `ntrees`: A nonnegative integer that determines the number of trees to grow.
*   `max_depth`: Maximum depth to grow the tree. A user-defined tuning parameter for controlling model complexity (by number of edges). Depth is the longest path from root to the furthest leaf. Maximum depth also specifies the maximum number of interactions that can be accounted for by the model.
*   `nbins_cats`: For categorical columns (factors), build a histogram of this many bins, then split at the best point. Higher values can lead to more overfitting. In our case, we set it equal to the number of stores we are trying to model (1,115).
*   `nfolds`: Number of folds for cross-validation. Since we are using stratified cross-validation, we have set this to 0.
*   `fold_column`: Column with cross-validation fold index assignment per observation, which we have set to `fold`, which was created in the “Create stratified folds for cross-validation” step above.

</div>

</div>

<div id="benchmarking-h20-and-random-forests" class="section level1">

## Benchmarking H20 and Random Forests

[![x-plot-time](http://2igww43ul7xe3nor5h2ah1yq.wpengine.netdna-cdn.com/wp-content/uploads/2015/12/x-plot-time.png)](http://2igww43ul7xe3nor5h2ah1yq.wpengine.netdna-cdn.com/wp-content/uploads/2015/12/x-plot-time.png)As noted before, H20 can use all the cores on a machine and hence should run substantially faster than if we used another random forest package in R. Szilard Pafka recently benchmarked several machine learning tools for scalability, speed and accuracy, including H2O. Pafka concluded that the H2O implementation of random forests is “fast, memory efficient and uses all cores. It deals with categorical variables automatically. It is also more accurate than R/Python, which may be because of dealing properly with the categorical variables, i.e. internally in the algo rather than working from a previously 1-hot encoded dataset (where the link between the dummies belonging to the same original variable is lost).”

- [Benchmarking Random Forest Implementations](http://datascience.la/benchmarking-random-forest-implementations/)

For more about information about this benchmarking study, see [Simple/limited/incomplete benchmark for scalability, speed and accuracy of machine learning libraries for classification](https://github.com/szilard/benchm-ml) and a video presentation, [Szilard Pafka: Benchmarking Machine Learning Tools for Scalability, Speed and Accuracy](https://vimeopro.com/eharmony/talks/video/132838730).

</div>

</div>

<div id="start-h20-cluster-with-all-available-threads" class="section level2">

<div id="benchmarking-h20-and-random-forests" class="section level1">

<div id="retrain-on-whole-training-set-without-the-validation-set." class="section level2">

## Retrain Random Forest On Training Set

We retrain on the whole training set (without the validation set or the cross-validation set) once we have tuned the parameters. Although not shown here, we extensively tested the different paramters for random forests as well as performed feature selection.

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;"># Use the whole train dataset.</span>
<span style="color: #888888;">trainHex<-as.h2o(train)</span>

<span style="color: #888888;"># Run Random Forest Model</span>
<span style="color: #888888;">rfHex <- h2o.randomForest(x=features,</span>
 <span style="color: #888888;">y="logSales",</span> 
 <span style="color: #888888;">model_id="introRF",</span>
 <span style="color: #888888;">training_frame=trainHex,</span>
 <span style="color: #888888;">mtries = -1, # default</span>
 <span style="color: #888888;">sample_rate = 0.632, # default</span>
 <span style="color: #888888;">ntrees = 100,</span>
 <span style="color: #888888;">max_depth = 30,</span>
 <span style="color: #888888;">nbins_cats = 1115, ## allow it to fit store ID</span>
 <span style="color: #888888;">nfolds = 0,</span>
 <span style="color: #888888;">seed = 12345678 #Seed for random numbers (affects sampling)</span>
 <span style="color: #888888;">)</span>

<span style="color: #888888;"># Model Summary and Variable Importanc</span>
<span style="color: #888888;">summary(rfHex)</span>
<span style="color: #888888;">varimps = data.frame(h2o.varimp(rfHex))</span>

<span style="color: #888888;">#Load test dataset into H2O from R</span>
<span style="color: #888888;">testHex<-as.h2o(test)</span>

<span style="color: #888888;">#Get predictions out; predicts in H2O, as.data.frame gets them into R</span>
<span style="color: #888888;">predictions<-as.data.frame(h2o.predict(rfHex,testHex))</span>

<span style="color: #888888;"># Return the predictions to the original scale of the Sales data</span>
<span style="color: #888888;">pred <- expm1(predictions[,1])</span>
<span style="color: #888888;">summary(pred)</span>
<span style="color: #888888;">submission <- data.frame(Id=test$Id, Sales=pred)</span>

<span style="color: #888888;"># Save the submission file</span>
<span style="color: #888888;">write.csv(submission, "../../data/H2O_Random_Forest_v47.csv",row.names=F)</span>
</pre>

</div>

</div>

<div id="gradient-boosted-models-in-a-nutshell" class="section level2">

# Gradient Boosted Models

We also utilized gradient boosted models (GBM) utilizing H2O as well.

#### Intuition

*   Average an ensemble of weakly predicting (small) trees where each tree “adjusts” to the “mistakes” of the preceding trees.
*   Boosting
*   Fits consecutive trees where each solves for the net error of the prior trees.

<div id="conceptual-1" class="section level4">

#### Conceptual

*   Boosting: ensemble of weak learners (the notion of “weak” is being challenged in practice)
*   Fits consecutive trees where each solves for the net loss of the prior trees
*   Results of new trees are applied partially to the entire solution.

</div>

<div id="practical-1" class="section level4">

<div id="strenths" class="section level5">

#### Strenths

*   Often best possible model
*   Robust
*   Directly optimizes cost function ##### Weaknesses
*   Overfits
*   Need to find proper stopping point
*   Sensitive to noise and extreme values
*   Several hyper-parameters
*   Lack of transparency

</div>

</div>

<div id="important-components" class="section level4">

#### Important components:

*   Number of trees
*   Maximum depth of tree
*   Learning rate (shrinkage parameter), where smaller learning rates tend to require larger number of tree and vice versa.

</div>

</div>

<div id="gradient-boosting-models-in-detail" class="section level2">

## Gradient Boosting Models in Detail

A GBM is an ensemble of either regression or classification tree models. Both are forward-learning ensemble methods that obtain predictive results using gradually improved estimations. Boosting is a flexible nonlinear regression procedure that helps improve the accuracy of trees. Weak classification algorithms are sequentially applied to the incrementally changed data to create a series of decision trees, producing an ensemble of weak prediction models. While boosting trees increases their accuracy, it also decreases speed and user interpretability. The gradient boosting method generalizes tree boosting to minimize these drawbacks. For more information, see [Gradient Boosted Models with H2O](http://h2o.ai/resources/)

</div>

<div id="h2os-gbm-functionalities" class="section level2">

## H2O’s GBM Functionalities

*   Supervised learning for regression and classification tasks
*   Distributed and parallelized computation on either a single node or a multi-node cluster
*   Fast and memory-e cient Java implementations of the algorithms
*   The ability to run H2O from R, Python, Scala, or the intuitive web UI (Flow)
*   automatic early stopping based on convergence of user-specified metrics to user-specified relative tolerance
*   stochastic gradient boosting with column and row sampling for better generalization
*   support for exponential families (Poisson, Gamma, Tweedie) and loss functions in addition to binomial (Bernoulli), Gaussian and multinomial distributions
*   grid search for hyperparameter optimization and model selection
*   modelexportinplainJavacodefordeploymentinproductionenvironments
*   additional parameters for model tuning (for a complete listing of parame- ters, refer to the Model Parameters section.)

</div>

<div id="key-parameters-for-gbm" class="section level2">

## Key Parameters for GBM

There are three primary paramters, or knobs, to adjust in order optimize GBMs.

1.  Adding trees will help. The default is 50.
2.  Increasing the learning rate will also help. The contribution of each tree will be stronger, so the model will move further away from the overall mean.
3.  Increasing the depth will help. This is the parameter that is the least straightforward. Tuning trees and learning rate both have direct impact that is easy to understand. Changing the depth means you are adjusting the “weakness” of each learner. Adding depth makes each tree fit the data closer.

</div>

<div id="get-a-summary-of-the-model-and-variable-importance-1" class="section level2">

## Retrain GBM On Training Set  

<div style="background: #f8f8f8; overflow: auto; width: auto; border: solid gray; border-width: .1em .1em .1em .8em; padding: .2em .6em;">

<pre style="margin: 0; line-height: 125%;"><span style="color: #888888;"># Train a GBM Model</span>
<span style="color: #888888;">gbmHex <- h2o.gbm(x=features,</span>
 <span style="color: #888888;">y="logSales",</span>
 <span style="color: #888888;">training_frame=trainHex,</span>
 <span style="color: #888888;">model_id="introGBM",</span>
 <span style="color: #888888;">nbins_cats=1115,</span>
 <span style="color: #888888;">sample_rate = 0.5,</span>
 <span style="color: #888888;">col_sample_rate = 0.5,</span>
 <span style="color: #888888;">max_depth = 20,</span>
 <span style="color: #888888;">learn_rate=0.05,</span>
 <span style="color: #888888;">seed = 12345678, #Seed for random numbers (affects sampling)</span>
 <span style="color: #888888;">ntrees = 250,</span>
 <span style="color: #888888;">fold_column="fold",</span>
 <span style="color: #888888;">validation_frame=validHex # validation set</span>
<span style="color: #888888;">)</span> 

<span style="color: #888888;">#Get a summary of the model and variable importance</span>
<span style="color: #888888;">summary(gbmHex)</span>
<span style="color: #888888;">varimps = data.frame(h2o.varimp(gbmHex))</span>

<span style="color: #888888;"># Get predictions out; predicts in H2O, as.data.frame gets them into R</span>
<span style="color: #888888;">predictions<-as.data.frame(h2o.predict(gbmHex,testHex))</span>

<span style="color: #888888;"># Return the predictions to the original scale of the Sales data</span>
<span style="color: #888888;">pred <- expm1(predictions[,1])</span>
<span style="color: #888888;">summary(pred)</span>
<span style="color: #888888;">submission <- data.frame(Id=test$Id, Sales=pred)</span>

<span style="color: #888888;"># Save the submission file</span>
<span style="color: #888888;">write.csv(submission, "../../data/H2O_GBM_v30.csv",row.names=F)</span>
</pre>

</div>

</div>

<div id="results" class="section level2">

## Results

Although we scored in top 5% of competition involving over 3,300 teams, the most valuable lesson learned was the understanding of accuracy vs interpretation trade off. By achieving the accuracy needed to score well for this Kaggle competition, we traded off interpretation that may have been needed to explain to management if this were to have been an actual business project. As a way of exploring this trade off, one of the first methods chosen was a multiple linear regression in order to gain a greater understanding of the characteristics for each feature. We achieved approximately 25% error using this simpler method. This may be an adequate prediction if market trends needed for broad scale decision making was the goal however being a Kaggle competition where accuracy was priority, this was not a method to be explored further.

</div>

</div>

</div>
