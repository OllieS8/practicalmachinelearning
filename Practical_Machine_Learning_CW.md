---
title: "Practical Machine Learning Coursework"
output: html_document
---

## Packages
Loading in the relevant packages, setting the seed for reproducibility and setting the working directory to where the training and testing csv files are downloaded.

```{r}
library(caret)
library(randomForest)
set.seed(1000)

# setting the working directory
setwd("~/R/Practical Machine Learning Course")
```
## Obtaining Data
Load in the training and testing data

```{r}
#reading in the training and testing data.
rawTrain <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

## Preprocessing

```{r}
# removing rows with new window = yes - not in test set
rawTrain <- subset(rawTrain, new_window=="no")

# partitioning training data into a training and cross validation set
inTrain <- createDataPartition(rawTrain$classe,p=0.5,list=FALSE)
training <- rawTrain[inTrain,]
crossVal <- rawTrain[-inTrain,]

# removing columns with near zero variance
ZeroVar <- nearZeroVar(training)
training <- training[,-ZeroVar]

#removing columns with lots of NA's and missing values
cntlength <- sapply(training, function(x) {
  sum(!(is.na(x) | x == ""))
})

nullcol <- names(cntlength[cntlength < 0.6 * length(training$classe)])

# removing descriptive fields
descriptcol <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                "cvtd_timestamp", "new_window", "num_window")
excludecols <- c(descriptcol ,nullcol)

training <- training[,!names(training) %in% excludecols]

```

## Training Model

```{r}
# creating model using random forest method. Reduced to 100 trees to speed up processing
modFit <- train(classe ~ ., method = "rf", data=training)#, ntree=100)
```

## Validating Model

### In Sample Validation

```{r}
# in sample error using training data
predTraining <- predict(modFit, training)
confusionMatrix(predTraining, training$classe)
```

### Out of Sample Validation

```{r}
# out of sample error using cross validation data
predVal <- predict(modFit, crossVal)
confusionMatrix(predVal, crossVal$classe)
```

## Predicting Test Set Classes

```{r}
# predictions on testing set
predTest <- predict(modFit, testing)
print(predTest)
```



