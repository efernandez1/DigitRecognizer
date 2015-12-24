setwd("~/Desktop/Projects/digitRecognizer")
.libPaths("~/Desktop/R_Packages/")

library("randomForest")
library("readr")
library("caret")
library("class")

train <- read_csv("../digitRecognizer/train.csv")
test <- read_csv("../digitRecognizer/test.csv")

## Normalize - could not tell a difference in models -----------
normalize <- function(x) {
    return( (x - min(x)) / (max(x) - min(x)) )
}

## Split data for testing -------------
TrainData <- train[1:5000,2:785]
TrainClass <- as.factor(train[1:5000,1])

TestData <- train[21001:30000,2:785]
TestClass <- as.factor(train[21001:30000,1])

TrainData <- as.data.frame(lapply(TrainData[,c(1:784)], normalize))
TrainData[is.na(TrainData)] <- 0

TestData <- as.data.frame(lapply(TestData[,c(1:784)], normalize))
TestData[is.na(TestData)] <- 0

## Remove columns with 0 variance - they tell no story -----------
badCols <- nearZeroVar(TrainData[,-1])
print(paste("Fraction of nearZeroVar columns:", round(length(badCols)/length(TrainData),4)))
TrainData <- TrainData[, -(badCols)]
TestData <- TestData[, -badCols]
## Removing columns does not improve the accuracy, but makes the model run faster ------- 


## Run Model -------------
ctrl <- trainControl(method="repeatedcv",repeats = 3)

knnFit <- train(TrainData, TrainClass,
                 method = "knn",
                 trControl = ctrl)

knnPred <- predict(knnFit, TestData)
confusionMatrix(knnPred, TestClass)
results <- data.frame(knnPred, TestClass)

results$correct <- as.factor(
    ifelse(results$knnPred == results$TestClass, "True", "False"))

summary(results$correct)

## Using KNN from the "class" package
knnModel <- knn(train=TrainData, test=TestData, cl=TrainClass, k=5)

## Prepare for submission-------------

submission$ImageId <- 1:28000
submission$Label <- knnModel

write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")








