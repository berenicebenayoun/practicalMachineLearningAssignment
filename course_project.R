setwd('/Users/benayoun/Dropbox/Coursera_data_science/practical_machine_learning/')
library('caret')
#Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

##What you should submit
#The goal of your project is to predict the manner in which they did the exercise. This is the 'classe' variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

#Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5.

#Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

my.har.training <- read.csv('pml-training.csv',header=TRUE)
my.har.testing <- read.csv('pml-testing.csv',header=TRUE)

summary(my.har.training)
# 160 variables

my.notUsable <- rep(FALSE,160)
for (i in 1:160) {
  if (sum(my.har.training[,i] %in% NA) > 19000) {
    my.notUsable[i] <- TRUE
  }
  if (sum(my.har.training[,i] %in% "") > 19000) {
    my.notUsable[i] <- TRUE
  }
  if (sum(my.har.training[,i] %in% "#DIV/0!") > 19000) {
    my.notUsable[i] <- TRUE
  }
}

sum(my.notUsable) # 100

summary(my.har.training[,-which(my.notUsable)])

###pairs(data.frame(my.har.training[,-which(my.notUsable)]))
# maybe correl, but keep for interpretability ?

my.clean.training <- data.frame(my.har.training[,-c(which(my.notUsable))])
colnames(my.clean.training)

my.clean.training <- my.clean.training[,-(1:5)]

# "raw_timestamp_part_1" "raw_timestamp_part_2" "cvtd_timestamp" -> not relevant
#1,2: X and people name should not be used for a general model

my.final.training <- createDataPartition(my.clean.training$classe, p=0.75, list=FALSE)
har.training <- my.clean.training[my.final.training,]
har.testing <- my.clean.training[-my.final.training,]

set.seed(123456)
my.ctrl.opt <-trainControl(method = "cv", number = 5)
my.rf.fit <-train(har.training$classe ~ ., method="rf",data=har.training,
                  importance=TRUE,trControl = my.ctrl.opt)
varImp(my.rf.fit, type=2)

my.rf.preds <- predict(my.rf.fit, har.testing)
my.confus.mat <- confusionMatrix(har.testing$classe, my.rf.preds)
my.confus.mat$overall
varImpPlot(my.rf.fit$finalModel, sort = TRUE, type = 1, pch = 16, bg = "red", cex = 1, main = "Variable importance")



######
my.har.testing <- read.csv('pml-testing.csv',header=TRUE)

my.clean.testing <- data.frame(my.har.testing[,-c(which(my.notUsable))])
my.clean.testing <- my.clean.testing[,-(1:5)]
my.test.preds <- predict(my.rf.fit, my.clean.testing)
my.test.preds <- as.character(my.test.preds)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(my.test.preds)