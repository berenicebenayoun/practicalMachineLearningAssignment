library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(975)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

bla<-1:length(mixtures$CompressiveStrength)

plot(bla,mixtures$CompressiveStrength)
points(bla[mixtures$Age < 5],mixtures$CompressiveStrength[mixtures$Age < 5],col="red",pch=16)
points(bla[mixtures$Age > 10],mixtures$CompressiveStrength[mixtures$Age >10],col="pink",pch=16)
points(bla[mixtures$Age > 30],mixtures$CompressiveStrength[mixtures$Age >30],col="blue",pch=16)
points(bla[mixtures$Age > 100],mixtures$CompressiveStrength[mixtures$Age >100],col="gold",pch=16)

plot(bla,mixtures$CompressiveStrength)
points(bla[mixtures$FlyAsh < 0.01],mixtures$CompressiveStrength[mixtures$FlyAsh < 0.01],col="red",pch=16)
points(bla[mixtures$FlyAsh > 0.04],mixtures$CompressiveStrength[mixtures$FlyAsh >0.04],col="pink",pch=16)
points(bla[mixtures$FlyAsh > 0.06],mixtures$CompressiveStrength[mixtures$FlyAsh >0.06],col="blue",pch=16)
points(bla[mixtures$FlyAsh > 0.08],mixtures$CompressiveStrength[mixtures$FlyAsh >0.08],col="gold",pch=16)


library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

colnames(adData[,58:69])

# perform PCA calculatioh
CR.pca <- prcomp(training[,58:69], scale = TRUE)
summary(CR.pca)

Create a training data set consisting of only the predictors with variable names beginning with IL and the diagnosis. 
Build two predictive models, one using the predictors as they are and one using PCA with principal components explaining 80% 
of the variance in the predictors. Use method="glm" in the train function. What is the accuracy of each method in the test set? 
Which is more accurate?

preProc <- preProcess(training[,58:69],method="pca",pcaComp=7)

training2 <- training[,58:69]
testing2 <- testing[,58:69]

fit1 <-train(training$diagnosis ~ .,
             method="glm",preProcess="pca",data=training2, trControl = trainControl(preProcOptions = list(thresh = 0.8)))

confusionMatrix(testing$diagnosis,predict(fit1,testing2))


Accuracy : 0.7195   


fit2 <-train(training$diagnosis ~ .,
             method="glm",data=training2)

confusionMatrix(testing$diagnosis,predict(fit2,testing2))

Accuracy : 0.6463 