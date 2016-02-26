# Loading the libraries
library(ggplot2) # library for plotting data
library(rpart) # library for decision tree model
library(randomForest) # library for random Forest model
library(Metrics) # this library contains the rmsle function, used to validate predictions on my train subsets

# Reading the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# First naive submission
sampleSubmission <- read.csv("sampleSubmission.csv")
firstSubmission <- sampleSubmission
firstSubmission$count <- mean(train$count)
write.csv(firstSubmission,file="firstSubmission.csv", row.names = FALSE)

# Making subsets of the train data for personal validation before submitting to Kaggle
trainIndices <- sample(1:length(train[,1]), size = length(train[,1])/2)
subTrain <- train[trainIndices,]
subTest <- train[-trainIndices,]

# Formatting the datetime variable on all sets
train$datetime <- as.POSIXct(train$datetime, format="%Y-%m-%d %H:%M:%S")
datetime = as.POSIXlt(train$datetime)
train$hour = datetime$hour
train$weekday = as.factor(datetime$wday)
train$month = as.factor(datetime$mon)
train$year = 1900 + datetime$year

test$datetime <- as.POSIXct(test$datetime, format="%Y-%m-%d %H:%M:%S")
datetime = as.POSIXlt(test$datetime)
test$hour = datetime$hour
test$weekday = as.factor(datetime$wday)
test$month = as.factor(datetime$mon)
test$year = 1900 + datetime$year

subTrain$datetime <- as.POSIXct(subTrain$datetime, format="%Y-%m-%d %H:%M:%S")
datetime = as.POSIXlt(subTrain$datetime)
subTrain$hour = datetime$hour
subTrain$weekday = as.factor(datetime$wday)
subTrain$month = as.factor(datetime$mon)
subTrain$year = 1900 + datetime$year

subTest$datetime <- as.POSIXct(subTest$datetime, format="%Y-%m-%d %H:%M:%S")
datetime = as.POSIXlt(subTest$datetime)
subTest$hour = datetime$hour
subTest$weekday = as.factor(datetime$wday)
subTest$month = as.factor(datetime$mon)
subTest$year = 1900 + datetime$year

# Plotting number of counts per hour, colored by weekdays, shows the importance of such split
ggplot(train, aes(x=hour, y=count, col=weekday)) + geom_point()

# All the variables are included in the formula, and will be applied on train sets with removed columns
fol <- formula(count ~ .)
# The removed columns are datetime, which is redundant with the 4 created variables,
# And also casual and registered which are not in the test dataset
# the control = rpart.control(cp = 0) attribute will often lead to overfitting but works well in our case
treeModel <- rpart(fol, data=train[,-c(1,10,11)],control = rpart.control(cp = 0))
treePrediction <- predict(treeModel, test)
forestModel <- randomForest(fol, data=train[,-c(1,10,11)])
forestPrediction <- predict(forestModel, test)

# Negative an NA values are forbidden but might happen in cases of regression
# Such cases will be rare, so turning them in 0 values will have no incidence on the result.
treePrediction[which(treePrediction < 0)] <- 0
treePrediction[which(is.na(treePrediction))] <- 0
forestPrediction[which(forestPrediction < 0)] <- 0
forestPrediction[which(is.na(forestPrediction))] <- 0

# Evaluation (to be used when the Model is used on SubTrain and the prediction on subTest)
#rmsle(subTest[,12],treePrediction)
#rmsle(subTest[,12],forestPrediction)

# Let's submit our models!
sampleSubmission <- read.csv("sampleSubmission.csv")
treeSubmission <- sampleSubmission
treeSubmission$count <- treePrediction
forestSubmission <- sampleSubmission
forestSubmission$count <- forestPrediction

# Writing Submission in file
write.csv(treeSubmission,file="treeSubmission.csv", row.names = FALSE)
write.csv(forestSubmission,file="forestSubmission.csv", row.names = FALSE)
