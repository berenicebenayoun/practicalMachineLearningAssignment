1. Subset the data to a training set and testing set based on the Case variable in the data set. 
2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings. 
3. In the final model what would be the final model prediction for cases with the following variable values:
  a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2 
b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100 
c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100 
d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2 

inTrain = which(segmentationOriginal$Case %in% "Train")
training = segmentationOriginal[ inTrain,]
testing = segmentationOriginal[-inTrain,]

set.seed(125)


2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings. 


fit1 <-train(training$Class ~ .,method="rpart",data=training)
plot(fit1$finalModel)

########################
library(pgmm)
data(olive)
olive = olive[,-1]


newdata = as.data.frame(t(colMeans(olive)))
fit1 <-train(olive$Area ~ .,method="tree",data=olive)
predict(fit1,newdata=newdata)

fit1 <-tree(olive$Area ~ .,data=olive)
predict(fit1,newdata=newdata)



library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

set.seed(13234)


Then set the seed to 13234 and fit a logistic regression model (method="glm", be sure to specify family="binomial") 
with Coronary Heart Disease (chd) as the outcome and age at onset, current alcohol consumption, obesity levels, cumulative tabacco,
type-A behavior, and low density lipoprotein cholesterol as predictors. Calculate the misclassification rate for your model 
using this function and a prediction on the "response" scale:

fit1 <-train(trainSA$chd ~ trainSA$age + trainSA$alcohol + trainSA$obesity + trainSA$tobacco  + trainSA$typea + trainSA$ldl,
             method="glm",family="binomial",data=trainSA)
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

#fit1 <-glm(trainSA$chd ~ trainSA$age + trainSA$alcohol + trainSA$obesity + trainSA$tobacco  + trainSA$typea + trainSA$ldl,
#            family="binomial",data=trainSA)


missClass(SAheart$chd,predict(fit1,SAheart))

missClass(trainSA$chd,predict(fit1,trainSA))
missClass(testSA$chd,predict(fit1,testSA))

library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 

set.seed(33833)
fit1 <-train(factor(vowel.train$y) ~ ., method="rf",data=vowel.train,importance=TRUE)
varImp(fit1, type=2 )

fit2 <-randomForest(vowel.train[,-1],factor(vowel.train$y)) 
importance(fit2)



set.seed(33833)
fit1 <-train(factor(vowel.train$y) ~ ., method="rf",data=vowel.train)
fit2 <-train(factor(vowel.train$y) ~ ., method="gbm",data=vowel.train)


confusionMatrix(factor(vowel.test$y),predict(fit1,vowel.test)) #0.6061   
confusionMatrix(factor(vowel.test$y),predict(fit2,vowel.test)) #

confusionMatrix(predict(fit1,vowel.test),predict(fit2,vowel.test)) #






######################################################
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]


set.seed(62433)

fit1 <-train(training$diagnosis ~ ., method="rf",data=training)
fit2 <-train(training$diagnosis ~ ., method="gbm",data=training,verbose=F)
fit3 <-train(training$diagnosis ~ ., method="lda",data=training)

pred1 <- predict(fit1,testing)
pred2 <- predict(fit2,testing)
pred3 <- predict(fit3,testing)

confusionMatrix(testing$diagnosis,pred1) #0.7683   
confusionMatrix(testing$diagnosis,pred2) #0.7927
confusionMatrix(testing$diagnosis,pred3) #0.7683


bla.data <- data.frame(pred1,pred2,pred3,diagnosis=testing$diagnosis)

combfit<- train(diagnosis ~., method="rf",data=bla.data)
combpred<- predict(combfit,bla.data)
confusionMatrix(testing$diagnosis,combpred) #0.8049


######################
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)



