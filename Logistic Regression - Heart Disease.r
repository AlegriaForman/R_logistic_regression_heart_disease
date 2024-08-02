# Install Libraries
print("This step will first install three R packages. Please wait until the packages are fully installed.")
print("Once the installation is complete, this step will print 'Installation complete!'")

install.packages("ResourceSelection")
install.packages("pROC")
install.packages("rpart.plot")

print("Installation complete!")

# Prepare Your Data Set
heart_data <- read.csv(file="heart_disease.csv", header=TRUE, sep=",")

# Converting appropriate variables to factors  
heart_data <- within(heart_data, {
   target <- factor(target)
   sex <- factor(sex)
   cp <- factor(cp)
   fbs <- factor(fbs)
   restecg <- factor(restecg)
   exang <- factor(exang)
   slope <- factor(slope)
   ca <- factor(ca)
   thal <- factor(thal)
})

head(heart_data, 10)

print("Number of variables")
ncol(heart_data)

print("Number of rows")
nrow(heart_data)

# First Logistic Regression Model

# This line will create the logistic model for hearth disease/target as predictor viariable of 
# age, trestbps, exang and thalach
hrtlogixreg <- glm(target ~ age + trestbps + exang + thalach, data = heart_data, family = "binomial")

summary(hrtlogixreg)

library(ResourceSelection)

print("Hosmer-Lemeshow Goodness of Fit Test")
hl = hoslem.test(hrtlogixreg$y, fitted(hrtlogixreg), g=50)
hl

# This line will perform the on Wald’s test
waldconf_int <- confint.default(hrtlogixreg, level=0.95)
round(waldconf_int,4)

# Predict yes or no heart disease for the data set using the model
default_heartmod1 <- heart_data[c('age', 'trestbps', 'exang', 'thalach')]
pred <- predict(hrtlogixreg, newdata=default_heartmod1, type='response')

# If the predicted probability of target is >=0.50 then predict heart disease 0 = no and 1 = yes 
depvar_pred = as.factor(ifelse(pred >= 0.5, '1', '0'))

# This creates the confusion matrix
conf.matrix <- table(heart_data$target, depvar_pred)[c('0','1'),c('0','1')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": target = ")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": target = ")

# Print nicely formatted confusion matrix
print("Confusion Matrix")
format(conf.matrix,justify="centre",digit=2)

library(pROC)

labels <- heart_data$target
predictions <- hrtlogixreg$fitted.values

roc <- roc(labels ~ predictions)

print("Area Under the Curve (AUC)")
round(auc(roc),4)

print("ROC Curve")
# True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
plot(roc, legacy.axes = TRUE)

print("Prediction: an individual having heart disease (age='50'), trestbps is 122 (trestbps=122), exang is yes (exang=1), thalach is 140 (thalach=140)")
hrtdata1 <- data.frame(age=50, trestbps=122, exang='1', thalach=140)
hrtpred1 <- predict(hrtlogixreg, hrtdata1, type='response')
round(hrtpred1, 4)

print("Prediction:  individual having heart disease (age='50'), trestbps is 130 (trestbps=130), exang is no (exang=0), thalach is 165 (thalach=165)")
hrtdata2 <- data.frame(age=50, trestbps=130, exang='0', thalach=165)
hrtpred2 <- predict(hrtlogixreg, hrtdata2, type='response')
round(hrtpred2, 4)

# Second Logistic Regression Model
# This line will create the 4. Model #2 – Second Logistic Regression Model for hearth disease/target as predictor  
# viariable of age, trestbps, cp and thalach
hrtlogixreg2 <- glm(target ~ age + trestbps + cp + thalach + I(age^2) + age:thalach, data = heart_data, family = "binomial")

summary(hrtlogixreg2)

library(ResourceSelection)

print("Hosmer-Lemeshow Goodness of Fit Test")
hl = hoslem.test(hrtlogixreg2$y, fitted(hrtlogixreg2), g=50)
hl

# This line will perform the on Wald’s test
waldconf_int2 <- confint.default(hrtlogixreg2, level=0.95)
round(waldconf_int2,4)

# Predict yes or no heart disease for the data set using the model
default_heartmod2 <- heart_data[c('age', 'trestbps', 'cp', 'thalach')]
pred2 <- predict(hrtlogixreg2, newdata=default_heartmod2, type='response')

# If the predicted probability of target is >=0.50 then predict heart disease 0 = no and 1 = yes 
depvar_pred2 = as.factor(ifelse(pred2 >= 0.5, '1', '0'))

# This creates the confusion matrix
conf.matrix <- table(heart_data$target, depvar_pred2)[c('0','1'),c('0','1')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": target = ")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": target = ")

# Print nicely formatted confusion matrix
print("Confusion Matrix")
format(conf.matrix,justify="centre",digit=2)

library(pROC)

labels <- heart_data$target
predictions2 <- hrtlogixreg2$fitted.values

roc <- roc(labels ~ predictions2)

print("Area Under the Curve (AUC)")
round(auc(roc),4)

print("ROC Curve")
# True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
plot(roc, legacy.axes = TRUE)

print("Prediction: an individual having heart disease (age='50'), trestbps is 115 (trestbps=115), cp is 0 (cp='0'), thalach is 133 (thalach=133)")
hrtdata3 <- data.frame(age=50, trestbps=115, cp='0', thalach=133)
hrtpred3 <- predict(hrtlogixreg2, hrtdata3, type='response')
round(hrtpred3, 4)

print("Prediction:  individual having heart disease (age='50'), trestbps is 125 (trestbps=125), cp is 1 (cp='1'), thalach is 155 (thalach=155)")
hrtdata4 <- data.frame(age=50, trestbps=125, cp='1', thalach=155)
hrtpred4 <- predict(hrtlogixreg2, hrtdata4, type='response')
round(hrtpred4, 4)

# Random Forest Classification Model

# 5. Random Forest Classification Model
set.seed(6522048)
library(randomForest)

# Partition the data set into training and testing data
samp.size = floor(0.85*nrow(heart_data))

# This line is re total rows in the training set
print("Number of rows for the training set")
train_ind = sample(seq_len(nrow(heart_data)), size = samp.size)
train.data = heart_data[train_ind,]
nrow(train.data)

# This line is re total rows in the testing set 
print("Number of rows for the testing set")
test.data = heart_data[-train_ind,]
nrow(test.data)

# This line is re total rows in the credit_default.csv
print("Number of rows in the heart_data.csv")
hrtdatatot = heart_data
nrow(hrtdatatot)

set.seed(6522048)
library(randomForest)

# checking
#=====================================================================
train = c()
test = c()
trees = c()

for(i in seq(from=1, to=150, by=1)) {
    #print(i)
    
    trees <- c(trees, i)
    
    model_rf9 <- randomForest(target ~ age+sex+cp+trestbps+chol+restecg+exang+ca, data=train.data, ntree = i)
    
    train.data.predix <- predict(model_rf9, train.data, type = "class")
    conf.matrix1 <- table(train.data$target, train.data.predix)
    train_error = 1-(sum(diag(conf.matrix1)))/sum(conf.matrix1)
    train <- c(train, train_error)
    
    test.data.predix <- predict(model_rf9, test.data, type = "class")
    conf.matrix2 <- table(test.data$target, test.data.predix)
    test_error = 1-(sum(diag(conf.matrix2)))/sum(conf.matrix2)
    test <- c(test, test_error)
}
 
#matplot (trees, cbind (train, test), ylim=c(0,0.5) , type = c("l", "l"), lwd=2, col=c("red","blue"), ylab="Error", xlab="number of trees")
#legend('topright',legend = c('training set','testing set'), col = c("red","blue"), lwd = 2 )

plot(trees, train,type = "l",ylim=c(0,1.0),col = "red", xlab = "Number of Trees", ylab = "Classification Error")
lines(test, type = "l", col = "blue")
legend('topright',legend = c('training set','testing set'), col = c("red","blue"), lwd = 2 )

set.seed(6522048)
library(randomForest)

model_rf9 <- randomForest(target ~ age+sex+cp+trestbps+chol+restecg+exang+ca, data=train.data, ntree = 25)

# Confusion matrix
print("===========================================================")
print('Confusion Matrix: TRAINING set based on random forest model built using 25 trees')
train.data.predict <- predict(model_rf9, train.data, type = "class")

# Construct the confusion matrix
conf.matrix <- table(train.data$target, train.data.predict)[,c('0','1')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": ")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": ")

# Print nicely formatted confusion matrix
format(conf.matrix,justify="centre",digit=2)


print("===========================================================")
print('Confusion Matrix: TESTING set based on random forest model built using 25 trees')
test.data.predict <- predict(model_rf9, test.data, type = "class")

# Construct the confusion matrix
conf.matrix <- table(test.data$target, test.data.predict)[,c('0','1')]
rownames(conf.matrix) <- paste("Actual", rownames(conf.matrix), sep = ": ")
colnames(conf.matrix) <- paste("Prediction", colnames(conf.matrix), sep = ": ")

# Print nicely formatted confusion matrix
format(conf.matrix,justify="centre",digit=2)

# Random Forest Regression Model

# 6.Random Forest Regression Model
set.seed(6522048)
library(randomForest)

# Partition the data set into training and testing data
samp.size = floor(0.80*nrow(heart_data))

# This line is re total rows in the training set
print("Number of rows for the training set")
train_ind1 = sample(seq_len(nrow(heart_data)), size = samp.size)
train.data1 = heart_data[train_ind1,]
nrow(train.data1)

# This line is re total rows in the testing set 
print("Number of rows for the testing set")
test.data1 = heart_data[-train_ind1,]
nrow(test.data1)

# This line is re total rows in the credit_default.csv
print("Number of rows in the heart_data.csv")
hrtdatatot1 = heart_data
nrow(hrtdatatot1)

set.seed(6522048)
library(randomForest)

# Root Mean Squared Error
RMSE = function(pred, obs) {
    return(sqrt( sum( (pred - obs)^2 )/length(pred) ) )
}

# Processing
#======================================
train = c()
test = c()
trees = c()

for(i in seq(from=1, to=80, by=1)) {
    
    trees <- c(trees, i)
    model_rf9a <- randomForest(thalach ~ age+sex+cp+trestbps+chol+restecg+exang+ca, data=train.data, ntree = i)
    
    pred <- predict(model_rf9a, newdata=train.data, type='response')
    rmse_train <-  RMSE(pred, train.data$thalach)
    train <- c(train, rmse_train)
    
    pred <- predict(model_rf9a, newdata=test.data, type='response')
    rmse_test <-  RMSE(pred, test.data$thalach)
    test <- c(test, rmse_test)
}
 
plot(trees, train,type = "l",col = "red", ylim=c(0,40), xlab = "Number of Trees", ylab = "Root Mean Squared Error")
lines(test, type = "l", col = "blue")
legend('topright',legend = c('training set','testing set'), col = c("red","blue"), lwd = 2 )

set.seed(6522048)
library(randomForest)

model_rf9b <- randomForest(thalach ~ age+sex+cp+trestbps+chol+restecg+exang+ca, data=train.data, ntree = 20)

# This is the line of the Root Mean Squared Error
RMSE = function(pred, obs) {
    return(sqrt( sum( (pred - obs)^2 )/length(pred) ) )
}

print("=====================================")
print('Root Mean Squared Error: TRAINING set based on Random Forest model built using 20 tree')
pred <- predict(model_rf9b, newdata=train.data, type='response')
round(RMSE(pred, train.data$thalach),4)


print("=====================================")
print('Root Mean Squared Error: TESTING set based on Random Forest model built using 20 tree')
pred <- predict(model_rf9b, newdata=test.data, type='response')
round(RMSE(pred, test.data$thalach),4)