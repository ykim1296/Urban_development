############# Fit random forest/ logistic regression/ extreme gradient boosting/ neural networks over 2015-2019 data #############
##### Install libraries
library(ggplot2)
library(dplyr)
library(randomForest)
require(xgboost)
library(ggpubr)
library(cowplot)
library(tensorflow)
library(keras)

##### Read data files
fld <- read.csv("C:/Users/ctwky/Dropbox (UFL)/Spatio_Temporal_Modeling_Urban_Development/Florida_Land_Development_1900_2019/7_category/Florida_Land_Development_1900_2019.csv")
dBlock <- read.csv("C:/Users/ctwky/Dropbox (UFL)/Spatio_Temporal_Modeling_Urban_Development/Florida_Land_Development_1900_2019/7_category/Distances_between_Block_Groups.csv")

##### Create response variable (the number of vacancy) and spatio-temporal variables
years.target <- seq(2015, 2019)
Lag <- 5
J <- 2
years <- seq(min(years.target - (Lag * (J + 3))), max(years.target), 1)  

fld.ys <- fld[which(fld$YEAR %in% years),]
fld.ys <- fld.ys[order(fld.ys$BLOCKID, fld.ys$YEAR),]


fld.ys$Y <- -c(rep(1, Lag), diff(fld.ys$NVACANT, lag = Lag))/c(rep(1, Lag), fld.ys$NVACANT[-c((dim(fld.ys)[1] - (Lag - 1)):(dim(fld.ys)[1]))] + 0.001)
fld.ys$L_J2 <- -c(rep(1, (Lag * J)), diff(fld.ys$NVACANT, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * J)), fld.ys$NVACANT[-c((dim(fld.ys)[1] - (Lag * J - 1)):(dim(fld.ys)[1]))] + 0.001)
fld.ys$L_J3 <- -c(rep(1, (Lag * (J + 1))), diff(fld.ys$NVACANT, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * (J + 1))), fld.ys$NVACANT[-c((dim(fld.ys)[1] - (Lag * (J + 1) - 1)):(dim(fld.ys)[1]))]+ 0.001)
fld.ys$L_J4 <- -c(rep(1, (Lag * (J + 2))), diff(fld.ys$NVACANT, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * (J + 2))), fld.ys$NVACANT[-c((dim(fld.ys)[1] - (Lag * (J + 2) - 1)):(dim(fld.ys)[1]))]+ 0.001)
fld.ys$L_J5 <- -c(rep(1, (Lag * (J + 3))), diff(fld.ys$NVACANT, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * (J + 3))), fld.ys$NVACANT[-c((dim(fld.ys)[1] - (Lag * (J + 3) - 1)):(dim(fld.ys)[1]))]+ 0.001)


nbh.size <- 10
block.nbhs <- t(apply(dBlock[,-1], 1, function(x) which(x %in% sort(x)[2:(nbh.size + 1)]) - 1))


fld.ys.nbh <- list()
k <- 1
for (yr in years) {
  df <- fld.ys[which(fld.ys$YEAR == yr), ]
  for (i in 1: dim(df)[1]) {
    df$NEFFECT[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "Y"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J2[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J2"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J3[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J3"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J4[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J4"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J5[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J5"] + 1), na.rm = T)
  }
  fld.ys.nbh[[k]] <- df
  k <- k + 1
}


fld.ys <- do.call(rbind, fld.ys.nbh)
fld.ys <- fld.ys[order(fld.ys$BLOCKID, fld.ys$YEAR),]



##### Set up a set of independent variables and response variable. Choose option 1 - option 4
############### Option 1: temporal covariates
var.list <- c("Y", 
              "L_J2", "L_J3", 
              "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3")

var.list.NoY <- c("L_J2", "L_J3", 
                  "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3")


############### Option 2: spatio-temporal covariates
var.list <- c("Y", 
              "L_J2", "L_J3", "L_J4", "L_J5", 
              "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")

var.list.NoY <- c("L_J2", "L_J3", "L_J4", "L_J5", 
                  "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")


############### Option 3: distance-related covariates
var.list <- c("Y", 
              "DRECREATION", "DSTORES", "DSUPERMARKET", "DREGIONALMALL", "DCOMMUNITYMALL", "DONEOFFICE", "DMULTIOFFICE", "DPROSERVICE", "DTRANSPORT", "DRESTAURANT", "DDRIVEINREST", "DFINANCIAL", "DINSURANCE", "DOTHERCOMMERCIAL", "DOTHERSERVICE", "DWHOLESALE", "DENTERNAINTMENT", "DHOTEL", "DLIGHTINDUSTIAL", "DHEAVYINDUSTRIAL", "DINDUSTRIAL", "DAGRICULTURAL", "DINSTITUTIONAL", "DEDUCATION", "DMILITARY", "DOPENSPACE", "DHOSPITALS", "DGOVERNMENT")

var.list.NoY <- c("DRECREATION", "DSTORES", "DSUPERMARKET", "DREGIONALMALL", "DCOMMUNITYMALL", "DONEOFFICE", "DMULTIOFFICE", "DPROSERVICE", "DTRANSPORT", "DRESTAURANT", "DDRIVEINREST", "DFINANCIAL", "DINSURANCE", "DOTHERCOMMERCIAL", "DOTHERSERVICE", "DWHOLESALE", "DENTERNAINTMENT", "DHOTEL", "DLIGHTINDUSTIAL", "DHEAVYINDUSTRIAL", "DINDUSTRIAL", "DAGRICULTURAL", "DINSTITUTIONAL", "DEDUCATION", "DMILITARY", "DOPENSPACE", "DHOSPITALS", "DGOVERNMENT")


############### Option 4: distance-related + spatio-temporal covariates
var.list <- c("Y", 
              "L_J2", "L_J3", "L_J4", "L_J5",
              "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5",
              "DRECREATION", "DSTORES", "DSUPERMARKET", "DREGIONALMALL", "DCOMMUNITYMALL", "DONEOFFICE", "DMULTIOFFICE", "DPROSERVICE", "DTRANSPORT", "DRESTAURANT", "DDRIVEINREST", "DFINANCIAL", "DINSURANCE", "DOTHERCOMMERCIAL", "DOTHERSERVICE", "DWHOLESALE", "DENTERNAINTMENT", "DHOTEL", "DLIGHTINDUSTIAL", "DHEAVYINDUSTRIAL", "DINDUSTRIAL", "DAGRICULTURAL", "DINSTITUTIONAL", "DEDUCATION", "DMILITARY", "DOPENSPACE", "DHOSPITALS", "DGOVERNMENT")

var.list.NoY <- c("L_J2", "L_J3", "L_J4", "L_J5", 
                  "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5", 
                  "DRECREATION", "DSTORES", "DSUPERMARKET", "DREGIONALMALL", "DCOMMUNITYMALL", "DONEOFFICE", "DMULTIOFFICE", "DPROSERVICE", "DTRANSPORT", "DRESTAURANT", "DDRIVEINREST", "DFINANCIAL", "DINSURANCE", "DOTHERCOMMERCIAL", "DOTHERSERVICE", "DWHOLESALE", "DENTERNAINTMENT", "DHOTEL", "DLIGHTINDUSTIAL", "DHEAVYINDUSTRIAL", "DINDUSTRIAL", "DAGRICULTURAL", "DINSTITUTIONAL", "DEDUCATION", "DMILITARY", "DOPENSPACE", "DHOSPITALS", "DGOVERNMENT")



##### Create index for training and test set
Years <- sort(seq(2015, 2019, 1), decreasing = T)

set.seed(59876)
index.test <- sample(seq(1, 11394), size = 0.2*11394)  # 11394: the number of observation in a single year
index.test.overall <- index.test
for (i in 1: (length(Years) - 1)) {
  index.test.overall <- c(index.test.overall, index.test + 11394 * i)
}



##### Transform variables 
Data <- fld.ys[which(fld.ys$YEAR %in% Years), which(names(fld.ys) %in% c(var.list, "BLOCKID", "YEAR"))]
Data <- Data[complete.cases(Data),]
Data <- Data[!is.infinite(rowSums(Data)),]
dim(Data)  

Data <- Data[order(Data$YEAR, decreasing = T), -which(colnames(Data) %in% c("BLOCKID", "YEAR"))]

##### Transform Y (log & standardized)
Data$Y.trans <- log(Data$Y + 1)
Data$Y.norm <- (Data$Y.trans - mean(Data$Y.trans, na.rm = T))/sd(Data$Y.trans, na.rm = T)

## Create binary response variable
ths <- median(Data$Y.norm)
Data$Y.norm.bin <- ifelse(Data$Y.norm < ths, 0, 1)    

##### Standardize covariates
### L_J2, L_J3 and other covariates
scaled.Data.1.1 <- scale(Data[which(names(Data) %in% var.list[-which(var.list %in% c("Y", "Y.trans", "Y.norm", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5", "L_J2", "L_J3", "L_J4", "L_J5"))])])
scaled.Data.1.2 <- scale(log(Data[which(names(Data) %in% var.list[which(var.list %in% c("L_J2", "L_J3", "L_J4", "L_J5"))])] + 1))
scaled.Data.1 <- cbind(scaled.Data.1.1, scaled.Data.1.2)

### NEFFECT.PAST.L_J2, NEFFECT.PAST.L_J3
scaled.Data.2 <- scale(Data[which(names(Data) %in% c("NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5"))])

Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- as.factor(Data$Y.norm.bin)
plot(Data.norm$Y.norm.bin, main = "Histogram of Y.norm.bin", xlab = "Y.norm.bin")

Data.norm <- Data.norm[complete.cases(Data.norm),]







############# Random forest #############
## Set training and test data set
test.data<- Data.norm[index.test.overall,]
train.data <- Data.norm[-index.test.overall,]

## Fit a random forest model
set.seed(822154)
train.data.rf <- randomForest(Y.norm.bin ~ ., data = train.data, ntree = 1000, mtry = 4, importance = T, nodesize = 3)
predict.rf.test <- predict(train.data.rf, test.data[,-which(names(test.data) == "Y.norm.bin")], type = "class")
predict.rf.train <- predict(train.data.rf, train.data[,-which(names(train.data) == "Y.norm.bin")], type = "class")
confusion.table.test <- table(test.data[,"Y.norm.bin"], predict.rf.test)
accuracy.test.rf <- sum(diag(confusion.table.test))/sum(confusion.table.test)
SE.pred <- sqrt(accuracy.test.rf*(1-accuracy.test.rf)/sum(confusion.table.test))


## Get a 95% CI 
paste0(round(accuracy.test.rf, 4), " " , "(", round(1.96*SE.pred, 4), ")")


## Get a variance importance plot
importance(train.data.rf)
imp <- varImpPlot(train.data.rf)

imp <- as.data.frame(imp)
imp$varnames <- rownames(imp)

INP <- ggplot(imp, aes(x=reorder(varnames, MeanDecreaseAccuracy), y=MeanDecreaseAccuracy)) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=MeanDecreaseAccuracy)) +
  ylab("MeanDecreaseAccuracy") +
  xlab("Variable Name") +
  coord_flip()

MSE <- ggplot(imp, aes(x=reorder(varnames, MeanDecreaseGini), y=MeanDecreaseGini)) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=MeanDecreaseGini)) +
  ylab("MeanDecreaseGini") +
  xlab("Variable Name") +
  coord_flip()

figure <- ggarrange(INP, MSE, labels = c("2015-2019", "2015-2019"),  ncol = 2)
figure






############# Logistic regression #############
## Set training and test data set
test.data<- Data.norm[index.test.overall,]
train.data <- Data.norm[-index.test.overall,]

## Fit logistic regression
train.data.lg <- glm(Y.norm.bin ~ ., data = train.data, family = binomial())
predict.lg.test <- ifelse(predict(train.data.lg, test.data[,-which(names(test.data) == "Y.norm.bin")], type = "response") > 0.5, 1, 0)
confusion.table.test.lg <- table(test.data[,"Y.norm.bin"], predict.lg.test)
accuracy.test.lg <- sum(diag(confusion.table.test.lg))/sum(confusion.table.test.lg)
SE.pred <- sqrt(accuracy.test.lg*(1-accuracy.test.lg)/sum(confusion.table.test.lg))

## Get a 95% CI 
paste0(round(accuracy.test.lg, 4), " " , "(", round(1.96*SE.pred, 4), ")")







############# Extreme gradient boosting #############
Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- Data$Y.norm.bin
Data.norm <- Data.norm[complete.cases(Data.norm),]
varLength <- dim(Data.norm)[2]

## Set training and test data set
Xs <- Data.norm[,1:(varLength-1)]
label <- Data.norm[,varLength]
test.Xs <- as.matrix(Xs[index.test.overall,])
test.label <- label[index.test.overall]
train.Xs <- as.matrix(Xs[-index.test.overall,])
train.label <- label[-index.test.overall]

## Transform the two data sets into xgb.Matrix
train = xgb.DMatrix(data=train.Xs, label=train.label)
test = xgb.DMatrix(data=test.Xs,label=test.label)

## Define the parameters for binary classification
params = list(
  booster="gbtree",
  eta=0.01,  
  max_depth=15,
  gamma=3,  
  subsample=0.9,
  colsample_bytree=1,
  lambda = 0,
  objective="binary:logistic",
  eval_metric="error"
)

## Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=train,
  nrounds=300,
  early_stopping_rounds=20,
  watchlist=list(val1=train,val2=test),  
  verbose=0
)

## Predict outcomes with the test data
xgb.pred = predict(xgb.fit, test.Xs, reshape=T)
xgb.pred = as.data.frame(xgb.pred)

predict.xgb.test <- ifelse(xgb.pred > 0.5, 1, 0)
confusion.table.test.xgb <- table(test.label, predict.xgb.test)
accuracy.test.xgb <- sum(diag(confusion.table.test.xgb))/sum(confusion.table.test.xgb)
SE.pred <- sqrt(accuracy.test.xgb*(1-accuracy.test.xgb)/sum(confusion.table.test.xgb))

## Get a 95% CI 
paste0(round(accuracy.test.xgb, 4), " " , "(", round(1.96*SE.pred, 4), ")")

## Get a variance importance plot
imp <- xgb.importance(model = xgb.fit)
imp <- as.data.frame(imp)
rownames(imp) <- imp$Feature 

Gain <- ggplot(imp, aes(x=reorder(Feature, Gain), y=Gain)) + 
  geom_point() +
  geom_segment(aes(x=Feature,xend=Feature,y=0,yend=Gain)) +
  ylab("Gain") +
  xlab("Variable Name") +
  coord_flip()

figure <- ggarrange(Gain, labels = "2015-2019",  ncol = 1)
figure








############# Neural network #############
Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- Data$Y.norm.bin
Data.norm <- Data.norm[complete.cases(Data.norm),]

test.data <- Data.norm[index.test.overall, which(names(Data.norm) != "Y.norm.bin")]
train.data <- Data.norm[-index.test.overall, which(names(Data.norm) != "Y.norm.bin")]
test.target <- Data.norm[index.test.overall, "Y.norm.bin"]
train.target <- Data.norm[-index.test.overall, "Y.norm.bin"]

### One hot encoding: redefining the dependent variable as a set of dummy variables
trainLabels <- to_categorical(train.target)
testLabels <- to_categorical(test.target)

input.shape <- dim(train.data)[2]

### Create the Model
model <- keras_model_sequential()

model %>%
  layer_dense(units=50, activation = 'relu', input_shape = input.shape) %>%     # this is for independent variables
  layer_dense(units=20, activation = 'relu') %>%
  layer_dense(units=2, activation = 'softmax')                        # this is for dependent variable. Represents the final output.

## Configure the model for the Learning process
model %>% compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')

## Fit the model
set.seed(75462)
index.val <- sample(1:nrow(train.data), size = 0.2*nrow(train.data))

x_val <- train.data[index.val,]
partial_x_train <- train.data[-index.val,]

y_val <- trainLabels[index.val,]
partial_y_train <- trainLabels[-index.val,]

train.data <- as.matrix(train.data)
test.data <- as.matrix(test.data)
trainLabels <- as.matrix(trainLabels)
testLabels <- as.matrix(testLabels)

history <- model %>%
  fit(train.data, # this is the input, the first 13 independent variables
      trainLabels,
      epochs=60,
      batch_size=512,
      validation_split = 0.1)

plot(history)

## Evaluate the model with test data
model%>% evaluate(test.data, testLabels)

## Look at the overall performance (confusion matrix)
prob<-model%>%
  predict_proba(test.data)

pred<-model%>%
  predict_classes(test.data)

pred.table <- table(Predicted = pred, Actual=Data.norm[index.test.overall, "Y.norm.bin"])
t(pred.table)
sum(diag(t(pred.table))/sum(t(pred.table)))

accuracy.test.nn <- sum(diag(t(pred.table))/sum(t(pred.table)))
SE.pred <- sqrt(accuracy.test.nn*(1-accuracy.test.nn)/sum(pred.table))

## Get a 95% CI 
paste0(round(accuracy.test.nn, 4), " " , "(", round(1.96*SE.pred, 4), ")")
```



############# Fit random forest/ logistic regression/ extreme gradient boosting/ neural networks over 2019 data to predict 2024 (The number of Vacancy) #############

##### Prepare data set to train RF/LR/EGB/NN with 8 spatio-temporal variables
############### Option 2: spatio-temporal covariates
var.list <- c("Y", 
              "L_J2", "L_J3", "L_J4", "L_J5", 
              "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")

var.list.NoY <- c("L_J2", "L_J3", "L_J4", "L_J5", 
                  "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")

Data <- fld.ys[which(fld.ys$YEAR %in% Years), which(names(fld.ys) %in% c(var.list, "BLOCKID", "YEAR"))]
Data <- Data[complete.cases(Data),]
Data <- Data[!is.infinite(rowSums(Data)),]
dim(Data)  

Data <- Data[order(Data$YEAR, decreasing = T), -which(colnames(Data) %in% c("BLOCKID", "YEAR"))]

##### Transform Y (log & standardized)
Data$Y.trans <- log(Data$Y + 1)
Data$Y.norm <- (Data$Y.trans - mean(Data$Y.trans, na.rm = T))/sd(Data$Y.trans, na.rm = T)

## Create binary response variable
ths <- median(Data$Y.norm)
Data$Y.norm.bin <- ifelse(Data$Y.norm < ths, 0, 1)    

##### Standardize covariates
### L_J2, L_J3 and other covariates
scaled.Data.1.1 <- scale(Data[which(names(Data) %in% var.list[-which(var.list %in% c("Y", "Y.trans", "Y.norm", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5", "L_J2", "L_J3", "L_J4", "L_J5"))])])
scaled.Data.1.2 <- scale(log(Data[which(names(Data) %in% var.list[which(var.list %in% c("L_J2", "L_J3", "L_J4", "L_J5"))])] + 1))
scaled.Data.1 <- cbind(scaled.Data.1.1, scaled.Data.1.2)

### NEFFECT.PAST.L_J2, NEFFECT.PAST.L_J3
scaled.Data.2 <- scale(Data[which(names(Data) %in% c("NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5"))])

Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- as.factor(Data$Y.norm.bin)
plot(Data.norm$Y.norm.bin, main = "Histogram of Y.norm.bin", xlab = "Y.norm.bin")

Data.norm <- Data.norm[complete.cases(Data.norm),]



##### Prepare data set to make prediction 
var.list.pred <- c("Y", "L_J2", "L_J3", "L_J4", 
                   "NEFFECT", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4")
Years <- 2019

Data.pred <- fld.ys[which(fld.ys$YEAR %in% Years), which(names(fld.ys) %in% c(var.list.pred, "BLOCKID", "YEAR"))]
Data.pred <- Data.pred[complete.cases(Data.pred),]
Data.pred <- Data.pred[!is.infinite(rowSums(Data.pred)),]

blockid <- Data.pred[order(Data.pred$YEAR, decreasing = T), "BLOCKID"]
Data.pred <- Data.pred[order(Data.pred$YEAR, decreasing = T), -which(colnames(Data.pred) %in% c("BLOCKID", "YEAR"))]


##### Standardize covariates
### L_J2, L_J3 and other covariates
scaled.Data.pred.1.1 <- scale(Data.pred[which(names(Data.pred) %in% var.list.pred[-which(var.list.pred %in% c("Y.trans", "Y.norm", "Y", "NEFFECT", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "L_J2", "L_J3", "L_J4"))])])
scaled.Data.pred.1.2 <- scale(log(Data.pred[which(names(Data.pred) %in% var.list.pred[which(var.list.pred %in% c("Y", "L_J2", "L_J3", "L_J4"))])] + 1))

scaled.Data.pred.1 <- cbind(scaled.Data.pred.1.1, scaled.Data.pred.1.2)


### NEFFECT.PAST.L_J2, NEFFECT.PAST.L_J3
scaled.Data.pred.2 <- scale(Data.pred[which(names(Data.pred) %in% c("NEFFECT", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4"))])


Data.pred.norm <- as.data.frame(cbind(blockid, scaled.Data.pred.1, scaled.Data.pred.2))
Data.pred.norm$Y.norm <- Data.pred$Y.norm
colnames(Data.pred.norm) <- c("BLOCKID", "L_J2", "L_J3", "L_J4", "L_J5", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")
dim(Data.pred.norm)



############# Random forest #############
## Set training and test data set
test.data<- Data.norm[index.test.overall,]
train.data <- Data.norm[-index.test.overall,]

## Fit a random forest model
set.seed(822154)
train.data.rf <- randomForest(Y.norm.bin ~ ., data = train.data, ntree = 1000, mtry = 4, importance = T, nodesize = 3)
predict.rf.test <- predict(train.data.rf, test.data[,-which(names(test.data) == "Y.norm.bin")], type = "class")
predict.rf.train <- predict(train.data.rf, train.data[,-which(names(train.data) == "Y.norm.bin")], type = "class")
confusion.table.test <- table(test.data[,"Y.norm.bin"], predict.rf.test)
accuracy.test.rf <- sum(diag(confusion.table.test))/sum(confusion.table.test)
SE.pred <- sqrt(accuracy.test.rf*(1-accuracy.test.rf)/sum(confusion.table.test))


## Get a 95% CI 
paste0(round(accuracy.test.rf, 4), " " , "(", round(1.96*SE.pred, 4), ")")


## Get a variance importance plot
importance(train.data.rf)
imp <- varImpPlot(train.data.rf)

imp <- as.data.frame(imp)
imp$varnames <- rownames(imp)

INP <- ggplot(imp, aes(x=reorder(varnames, MeanDecreaseAccuracy), y=MeanDecreaseAccuracy)) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=MeanDecreaseAccuracy)) +
  ylab("MeanDecreaseAccuracy") +
  xlab("Variable Name") +
  coord_flip()

MSE <- ggplot(imp, aes(x=reorder(varnames, MeanDecreaseGini), y=MeanDecreaseGini)) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=MeanDecreaseGini)) +
  ylab("MeanDecreaseGini") +
  xlab("Variable Name") +
  coord_flip()

figure <- ggarrange(INP, MSE, labels = c("2015-2019", "2015-2019"),  ncol = 2)
figure


##### Predict 2024
predict.rf.future.2024.NVACANT <- predict(train.data.rf, Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")], type = "class")










############# Logistic regression #############
## Set training and test data set
test.data<- Data.norm[index.test.overall,]
train.data <- Data.norm[-index.test.overall,]

## Fit logistic regression
train.data.lg <- glm(Y.norm.bin ~ ., data = train.data, family = binomial())
predict.lg.test <- ifelse(predict(train.data.lg, test.data[,-which(names(test.data) == "Y.norm.bin")], type = "response") > 0.5, 1, 0)
confusion.table.test.lg <- table(test.data[,"Y.norm.bin"], predict.lg.test)
accuracy.test.lg <- sum(diag(confusion.table.test.lg))/sum(confusion.table.test.lg)
SE.pred <- sqrt(accuracy.test.lg*(1-accuracy.test.lg)/sum(confusion.table.test.lg))

## Get a 95% CI 
paste0(round(accuracy.test.lg, 4), " " , "(", round(1.96*SE.pred, 4), ")")

##### Predict 2024
predict.lg.future.2024.NVACANT <- ifelse(predict(train.data.lg, Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")], type = "response") > 0.5, 1, 0)








############# Extreme gradient boosting #############
Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- Data$Y.norm.bin
Data.norm <- Data.norm[complete.cases(Data.norm),]
varLength <- dim(Data.norm)[2]

## Set training and test data set
Xs <- Data.norm[,1:(varLength-1)]
label <- Data.norm[,varLength]
test.Xs <- as.matrix(Xs[index.test.overall,])
test.label <- label[index.test.overall]
train.Xs <- as.matrix(Xs[-index.test.overall,])
train.label <- label[-index.test.overall]

## Transform the two data sets into xgb.Matrix
train = xgb.DMatrix(data=train.Xs, label=train.label)
test = xgb.DMatrix(data=test.Xs,label=test.label)

## Define the parameters for binary classification
params = list(
  booster="gbtree",
  eta=0.01,  
  max_depth=15,
  gamma=3,  
  subsample=0.9,
  colsample_bytree=1,
  lambda = 0,
  objective="binary:logistic",
  eval_metric="error"
)

## Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=train,
  nrounds=300,
  early_stopping_rounds=20,
  watchlist=list(val1=train,val2=test),  
  verbose=0
)

## Predict outcomes with the test data
xgb.pred = predict(xgb.fit, test.Xs, reshape=T)
xgb.pred = as.data.frame(xgb.pred)

predict.xgb.test <- ifelse(xgb.pred > 0.5, 1, 0)
confusion.table.test.xgb <- table(test.label, predict.xgb.test)
accuracy.test.xgb <- sum(diag(confusion.table.test.xgb))/sum(confusion.table.test.xgb)
SE.pred <- sqrt(accuracy.test.xgb*(1-accuracy.test.xgb)/sum(confusion.table.test.xgb))

## Get a 95% CI 
paste0(round(accuracy.test.xgb, 4), " " , "(", round(1.96*SE.pred, 4), ")")

## Get a variance importance plot
imp <- xgb.importance(model = xgb.fit)
imp <- as.data.frame(imp)
rownames(imp) <- imp$Feature 

Gain <- ggplot(imp, aes(x=reorder(Feature, Gain), y=Gain)) + 
  geom_point() +
  geom_segment(aes(x=Feature,xend=Feature,y=0,yend=Gain)) +
  ylab("Gain") +
  xlab("Variable Name") +
  coord_flip()

figure <- ggarrange(Gain, labels = "2015-2019",  ncol = 1)
figure

##### Predict 2024
xgb.pred.future = predict(xgb.fit, as.matrix(Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")]), reshape=T)
xgb.pred.future = as.data.frame(xgb.pred.future)

predict.xgb.future.2024.NVACANT <- ifelse(xgb.pred.future > 0.5, 1, 0)
```

**Neural network**
  
  ```{r}
Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- Data$Y.norm.bin
Data.norm <- Data.norm[complete.cases(Data.norm),]

test.data <- Data.norm[index.test.overall, which(names(Data.norm) != "Y.norm.bin")]
train.data <- Data.norm[-index.test.overall, which(names(Data.norm) != "Y.norm.bin")]
test.target <- Data.norm[index.test.overall, "Y.norm.bin"]
train.target <- Data.norm[-index.test.overall, "Y.norm.bin"]

### One hot encoding: redefining the dependent variable as a set of dummy variables
trainLabels <- to_categorical(train.target)
testLabels <- to_categorical(test.target)

input.shape <- dim(train.data)[2]

### Create the Model
model <- keras_model_sequential()

model %>%
  layer_dense(units=50, activation = 'relu', input_shape = input.shape) %>%     # this is for independent variables
  layer_dense(units=20, activation = 'relu') %>%
  layer_dense(units=2, activation = 'softmax')                        # this is for dependent variable. Represents the final output.

## Configure the model for the Learning process
model %>% compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')

## Fit the model
set.seed(75462)
index.val <- sample(1:nrow(train.data), size = 0.2*nrow(train.data))

x_val <- train.data[index.val,]
partial_x_train <- train.data[-index.val,]

y_val <- trainLabels[index.val,]
partial_y_train <- trainLabels[-index.val,]

train.data <- as.matrix(train.data)
test.data <- as.matrix(test.data)
trainLabels <- as.matrix(trainLabels)
testLabels <- as.matrix(testLabels)

history <- model %>%
  fit(train.data, # this is the input, the first 13 independent variables
      trainLabels,
      epochs=60,
      batch_size=512,
      validation_split = 0.1)

plot(history)

## Evaluate the model with test data
model%>% evaluate(test.data, testLabels)

## Look at the overall performance (confusion matrix)
prob<-model%>%
  predict_proba(test.data)

pred<-model%>%
  predict_classes(test.data)

pred.table <- table(Predicted = pred, Actual=Data.norm[index.test.overall, "Y.norm.bin"])
t(pred.table)
sum(diag(t(pred.table))/sum(t(pred.table)))

accuracy.test.nn <- sum(diag(t(pred.table))/sum(t(pred.table)))
SE.pred <- sqrt(accuracy.test.nn*(1-accuracy.test.nn)/sum(pred.table))

## Get a 95% CI 
paste0(round(accuracy.test.nn, 4), " " , "(", round(1.96*SE.pred, 4), ")")

##### Predict 2024
prob <- model%>%
  predict_proba(as.matrix(Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")]))

predict.nn.future.2024.NVACANT <- model%>%
  predict_classes(as.matrix(Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")]))













############# Fit random forest/ logistic regression/ extreme gradient boosting/ neural networks over 2019 data to predict 2024 (The number of single-family residential) #############

##### Prepare data set to train RF/LR/EGB/NN with 8 spatio-temporal variables
years.target <- seq(2015, 2019)
Lag <- 5
J <- 2
years <- seq(min(years.target - (Lag * (J + 3))), max(years.target), 1)  

fld.ys <- fld[which(fld$YEAR %in% years),]
fld.ys <- fld.ys[order(fld.ys$BLOCKID, fld.ys$YEAR),]


fld.ys$Y <- c(rep(1, Lag), diff(fld.ys$NSINGLERES, lag = Lag))/c(rep(1, Lag), fld.ys$NSINGLERES[-c((dim(fld.ys)[1] - (Lag - 1)):(dim(fld.ys)[1]))] + 0.001)
fld.ys$L_J2 <- c(rep(1, (Lag * J)), diff(fld.ys$NSINGLERES, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * J)), fld.ys$NSINGLERES[-c((dim(fld.ys)[1] - (Lag * J - 1)):(dim(fld.ys)[1]))] + 0.001)
fld.ys$L_J3 <- c(rep(1, (Lag * (J + 1))), diff(fld.ys$NSINGLERES, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * (J + 1))), fld.ys$NSINGLERES[-c((dim(fld.ys)[1] - (Lag * (J + 1) - 1)):(dim(fld.ys)[1]))]+ 0.001)
fld.ys$L_J4 <- c(rep(1, (Lag * (J + 2))), diff(fld.ys$NSINGLERES, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * (J + 2))), fld.ys$NSINGLERES[-c((dim(fld.ys)[1] - (Lag * (J + 2) - 1)):(dim(fld.ys)[1]))]+ 0.001)
fld.ys$L_J5 <- c(rep(1, (Lag * (J + 3))), diff(fld.ys$NSINGLERES, lag = Lag ))[1:dim(fld.ys)[1]]/c(rep(1, (Lag * (J + 3))), fld.ys$NSINGLERES[-c((dim(fld.ys)[1] - (Lag * (J + 3) - 1)):(dim(fld.ys)[1]))]+ 0.001)


fld.ys.nbh <- list()
k <- 1
for (yr in years) {
  df <- fld.ys[which(fld.ys$YEAR == yr), ]
  for (i in 1: dim(df)[1]) {
    df$NEFFECT[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "Y"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J2[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J2"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J3[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J3"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J4[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J4"] + 1), na.rm = T)
    df$NEFFECT.PAST.L_J5[i] <- mean(log(df[which(df$BLOCKID %in% block.nbhs[i, ]), "L_J5"] + 1), na.rm = T)
  }
  fld.ys.nbh[[k]] <- df
  k <- k + 1
}


fld.ys <- do.call(rbind, fld.ys.nbh)
fld.ys <- fld.ys[order(fld.ys$BLOCKID, fld.ys$YEAR),]



############### Option 2: spatio-temporal covariates
var.list <- c("Y", 
              "L_J2", "L_J3", "L_J4", "L_J5", 
              "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")

var.list.NoY <- c("L_J2", "L_J3", "L_J4", "L_J5", 
                  "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")

Data <- fld.ys[which(fld.ys$YEAR %in% Years), which(names(fld.ys) %in% c(var.list, "BLOCKID", "YEAR"))]
Data <- Data[complete.cases(Data),]
Data <- Data[!is.infinite(rowSums(Data)),]
dim(Data)  

Data <- Data[order(Data$YEAR, decreasing = T), -which(colnames(Data) %in% c("BLOCKID", "YEAR"))]

##### Transform Y (log & standardized)
Data$Y.trans <- log(Data$Y + 1)
Data$Y.norm <- (Data$Y.trans - mean(Data$Y.trans, na.rm = T))/sd(Data$Y.trans, na.rm = T)

## Create binary response variable
ths <- median(Data$Y.norm)
Data$Y.norm.bin <- ifelse(Data$Y.norm < ths, 0, 1)    

##### Standardize covariates
### L_J2, L_J3 and other covariates
scaled.Data.1.1 <- scale(Data[which(names(Data) %in% var.list[-which(var.list %in% c("Y", "Y.trans", "Y.norm", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5", "L_J2", "L_J3", "L_J4", "L_J5"))])])
scaled.Data.1.2 <- scale(log(Data[which(names(Data) %in% var.list[which(var.list %in% c("L_J2", "L_J3", "L_J4", "L_J5"))])] + 1))
scaled.Data.1 <- cbind(scaled.Data.1.1, scaled.Data.1.2)

### NEFFECT.PAST.L_J2, NEFFECT.PAST.L_J3
scaled.Data.2 <- scale(Data[which(names(Data) %in% c("NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5"))])

Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- as.factor(Data$Y.norm.bin)
plot(Data.norm$Y.norm.bin, main = "Histogram of Y.norm.bin", xlab = "Y.norm.bin")

Data.norm <- Data.norm[complete.cases(Data.norm),]




##### Prepare data set to make prediction 
var.list.pred <- c("Y", "L_J2", "L_J3", "L_J4", 
                   "NEFFECT", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4")
Years <- 2019

Data.pred <- fld.ys[which(fld.ys$YEAR %in% Years), which(names(fld.ys) %in% c(var.list.pred, "BLOCKID", "YEAR"))]
Data.pred <- Data.pred[complete.cases(Data.pred),]
Data.pred <- Data.pred[!is.infinite(rowSums(Data.pred)),]

blockid <- Data.pred[order(Data.pred$YEAR, decreasing = T), "BLOCKID"]
Data.pred <- Data.pred[order(Data.pred$YEAR, decreasing = T), -which(colnames(Data.pred) %in% c("BLOCKID", "YEAR"))]


##### Standardize covariates
### L_J2, L_J3 and other covariates
scaled.Data.pred.1.1 <- scale(Data.pred[which(names(Data.pred) %in% var.list.pred[-which(var.list.pred %in% c("Y.trans", "Y.norm", "Y", "NEFFECT", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "L_J2", "L_J3", "L_J4"))])])
scaled.Data.pred.1.2 <- scale(log(Data.pred[which(names(Data.pred) %in% var.list.pred[which(var.list.pred %in% c("Y", "L_J2", "L_J3", "L_J4"))])] + 1))

scaled.Data.pred.1 <- cbind(scaled.Data.pred.1.1, scaled.Data.pred.1.2)


### NEFFECT.PAST.L_J2, NEFFECT.PAST.L_J3
scaled.Data.pred.2 <- scale(Data.pred[which(names(Data.pred) %in% c("NEFFECT", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4"))])


Data.pred.norm <- as.data.frame(cbind(blockid, scaled.Data.pred.1, scaled.Data.pred.2))
Data.pred.norm$Y.norm <- Data.pred$Y.norm
colnames(Data.pred.norm) <- c("BLOCKID", "L_J2", "L_J3", "L_J4", "L_J5", "NEFFECT.PAST.L_J2", "NEFFECT.PAST.L_J3", "NEFFECT.PAST.L_J4", "NEFFECT.PAST.L_J5")
dim(Data.pred.norm)











############# Random forest #############
## Set training and test data set
test.data<- Data.norm[index.test.overall,]
train.data <- Data.norm[-index.test.overall,]

## Fit a random forest model
set.seed(822154)
train.data.rf <- randomForest(Y.norm.bin ~ ., data = train.data, ntree = 1000, mtry = 4, importance = T, nodesize = 3)
predict.rf.test <- predict(train.data.rf, test.data[,-which(names(test.data) == "Y.norm.bin")], type = "class")
predict.rf.train <- predict(train.data.rf, train.data[,-which(names(train.data) == "Y.norm.bin")], type = "class")
confusion.table.test <- table(test.data[,"Y.norm.bin"], predict.rf.test)
accuracy.test.rf <- sum(diag(confusion.table.test))/sum(confusion.table.test)
SE.pred <- sqrt(accuracy.test.rf*(1-accuracy.test.rf)/sum(confusion.table.test))


## Get a 95% CI 
paste0(round(accuracy.test.rf, 4), " " , "(", round(1.96*SE.pred, 4), ")")


## Get a variance importance plot
importance(train.data.rf)
imp <- varImpPlot(train.data.rf)

imp <- as.data.frame(imp)
imp$varnames <- rownames(imp)

INP <- ggplot(imp, aes(x=reorder(varnames, MeanDecreaseAccuracy), y=MeanDecreaseAccuracy)) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=MeanDecreaseAccuracy)) +
  ylab("MeanDecreaseAccuracy") +
  xlab("Variable Name") +
  coord_flip()

MSE <- ggplot(imp, aes(x=reorder(varnames, MeanDecreaseGini), y=MeanDecreaseGini)) + 
  geom_point() +
  geom_segment(aes(x=varnames,xend=varnames,y=0,yend=MeanDecreaseGini)) +
  ylab("MeanDecreaseGini") +
  xlab("Variable Name") +
  coord_flip()

figure <- ggarrange(INP, MSE, labels = c("2015-2019", "2015-2019"),  ncol = 2)
figure


##### Predict 2024
predict.rf.future.2024.NSINGLERES <- predict(train.data.rf, Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")], type = "class")










############# Logistic regression #############
## Set training and test data set
test.data<- Data.norm[index.test.overall,]
train.data <- Data.norm[-index.test.overall,]

## Fit logistic regression
train.data.lg <- glm(Y.norm.bin ~ ., data = train.data, family = binomial())
predict.lg.test <- ifelse(predict(train.data.lg, test.data[,-which(names(test.data) == "Y.norm.bin")], type = "response") > 0.5, 1, 0)
confusion.table.test.lg <- table(test.data[,"Y.norm.bin"], predict.lg.test)
accuracy.test.lg <- sum(diag(confusion.table.test.lg))/sum(confusion.table.test.lg)
SE.pred <- sqrt(accuracy.test.lg*(1-accuracy.test.lg)/sum(confusion.table.test.lg))

## Get a 95% CI 
paste0(round(accuracy.test.lg, 4), " " , "(", round(1.96*SE.pred, 4), ")")

##### Predict 2024
predict.lg.future.2024.NSINGLERES <- ifelse(predict(train.data.lg, Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")], type = "response") > 0.5, 1, 0)










############# Extreme gradient boosting #############
Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- Data$Y.norm.bin
Data.norm <- Data.norm[complete.cases(Data.norm),]
varLength <- dim(Data.norm)[2]

## Set training and test data set
Xs <- Data.norm[,1:(varLength-1)]
label <- Data.norm[,varLength]
test.Xs <- as.matrix(Xs[index.test.overall,])
test.label <- label[index.test.overall]
train.Xs <- as.matrix(Xs[-index.test.overall,])
train.label <- label[-index.test.overall]

## Transform the two data sets into xgb.Matrix
train = xgb.DMatrix(data=train.Xs, label=train.label)
test = xgb.DMatrix(data=test.Xs,label=test.label)

## Define the parameters for binary classification
params = list(
  booster="gbtree",
  eta=0.01,  
  max_depth=15,
  gamma=3,  
  subsample=0.9,
  colsample_bytree=1,
  lambda = 0,
  objective="binary:logistic",
  eval_metric="error"
)

## Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=train,
  nrounds=300,
  early_stopping_rounds=20,
  watchlist=list(val1=train,val2=test),  
  verbose=0
)

## Predict outcomes with the test data
xgb.pred = predict(xgb.fit, test.Xs, reshape=T)
xgb.pred = as.data.frame(xgb.pred)

predict.xgb.test <- ifelse(xgb.pred > 0.5, 1, 0)
confusion.table.test.xgb <- table(test.label, predict.xgb.test)
accuracy.test.xgb <- sum(diag(confusion.table.test.xgb))/sum(confusion.table.test.xgb)
SE.pred <- sqrt(accuracy.test.xgb*(1-accuracy.test.xgb)/sum(confusion.table.test.xgb))

## Get a 95% CI 
paste0(round(accuracy.test.xgb, 4), " " , "(", round(1.96*SE.pred, 4), ")")

## Get a variance importance plot
imp <- xgb.importance(model = xgb.fit)
imp <- as.data.frame(imp)
rownames(imp) <- imp$Feature 

Gain <- ggplot(imp, aes(x=reorder(Feature, Gain), y=Gain)) + 
  geom_point() +
  geom_segment(aes(x=Feature,xend=Feature,y=0,yend=Gain)) +
  ylab("Gain") +
  xlab("Variable Name") +
  coord_flip()

figure <- ggarrange(Gain, labels = "2015-2019",  ncol = 1)
figure

##### Predict 2024
xgb.pred.future = predict(xgb.fit, as.matrix(Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")]), reshape=T)
xgb.pred.future = as.data.frame(xgb.pred.future)

predict.xgb.future.2024.NSINGLERES <- ifelse(xgb.pred.future > 0.5, 1, 0)










############# Neural network #############
Data.norm <- as.data.frame(cbind(scaled.Data.1, scaled.Data.2))
Data.norm$Y.norm.bin <- Data$Y.norm.bin
Data.norm <- Data.norm[complete.cases(Data.norm),]

test.data <- Data.norm[index.test.overall, which(names(Data.norm) != "Y.norm.bin")]
train.data <- Data.norm[-index.test.overall, which(names(Data.norm) != "Y.norm.bin")]
test.target <- Data.norm[index.test.overall, "Y.norm.bin"]
train.target <- Data.norm[-index.test.overall, "Y.norm.bin"]

### One hot encoding: redefining the dependent variable as a set of dummy variables
trainLabels <- to_categorical(train.target)
testLabels <- to_categorical(test.target)

input.shape <- dim(train.data)[2]

### Create the Model
model <- keras_model_sequential()

model %>%
  layer_dense(units=50, activation = 'relu', input_shape = input.shape) %>%     # this is for independent variables
  layer_dense(units=20, activation = 'relu') %>%
  layer_dense(units=2, activation = 'softmax')                        # this is for dependent variable. Represents the final output.

## Configure the model for the Learning process
model %>% compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')

## Fit the model
set.seed(75462)
index.val <- sample(1:nrow(train.data), size = 0.2*nrow(train.data))

x_val <- train.data[index.val,]
partial_x_train <- train.data[-index.val,]

y_val <- trainLabels[index.val,]
partial_y_train <- trainLabels[-index.val,]

train.data <- as.matrix(train.data)
test.data <- as.matrix(test.data)
trainLabels <- as.matrix(trainLabels)
testLabels <- as.matrix(testLabels)

history <- model %>%
  fit(train.data, # this is the input, the first 13 independent variables
      trainLabels,
      epochs=60,
      batch_size=512,
      validation_split = 0.1)

plot(history)

## Evaluate the model with test data
model%>% evaluate(test.data, testLabels)

## Look at the overall performance (confusion matrix)
prob<-model%>%
  predict_proba(test.data)

pred<-model%>%
  predict_classes(test.data)

pred.table <- table(Predicted = pred, Actual=Data.norm[index.test.overall, "Y.norm.bin"])
t(pred.table)
sum(diag(t(pred.table))/sum(t(pred.table)))

accuracy.test.nn <- sum(diag(t(pred.table))/sum(t(pred.table)))
SE.pred <- sqrt(accuracy.test.nn*(1-accuracy.test.nn)/sum(pred.table))

## Get a 95% CI 
paste0(round(accuracy.test.nn, 4), " " , "(", round(1.96*SE.pred, 4), ")")

##### Predict 2024
prob <- model%>%
  predict_proba(as.matrix(Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")]))

predict.nn.future.2024.NSINGLERES <- model%>%
  predict_classes(as.matrix(Data.pred.norm[,-which(names(Data.pred.norm) == "BLOCKID")]))












############# Plot predicted values (2024 and 2029) #############
##### Read data sets containing latitude and longitude information
longlat <- read.csv("C:/Users/ctwky/Dropbox (UFL)/Spatio_Temporal_Modeling_Urban_Development/Florida_Land_Development_1900_2019/Florida_Block_Groups_Lat_Long.csv")
fl_big_cities <- read.csv("D:/Research/Dr.Safikhani/Urban Development/R codes/Florida_big_cities.csv")


##### Get Latitude and Longitude information
longlat2 <- fld[1:11394, c("LONGITUDE", "LATITUDE", "COUNTY")]

pred.RF <- as.data.frame(cbind(predict.rf.future.2024.NSINGLERES, predict.rf.future.2024.NVACANT) - 1)
pred.LR <- as.data.frame(cbind(predict.lg.future.2024.NSINGLERES, predict.lg.future.2024.NVACANT) - 1)
pred.NN <- as.data.frame(cbind(predict.nn.future.2024.NSINGLERES, predict.nn.future.2024.NVACANT) - 1)
pred.EGB <- as.data.frame(cbind(predict.xgb.future.2024.NSINGLERES, predict.xgb.future.2024.NVACANT) - 1)

pred.RF$Longitude <- longlat$LONG
pred.LR$Longitude <- longlat$LONG
pred.NN$Longitude <- longlat$LONG
pred.EGB$Longitude <- longlat$LONG

pred.RF$Latitude <- longlat$LAT
pred.LR$Latitude <- longlat$LAT
pred.NN$Latitude <- longlat$LAT
pred.EGB$Latitude <- longlat$LAT

pred.RF$County <- longlat2$COUNTY
pred.LR$County <- longlat$COUNTY
pred.NN$County <- longlat$COUNTY
pred.EGB$County <- longlat$COUNTY



##### Prepare background maps
states <- map_data("state")
fl_df <- subset(states, region == "florida")

counties <- map_data("county")
fl_county <- subset(counties, region == "florida")

fl_base <- ggplot(data = fl_df, mapping = aes(x = long, y = lat, group = group)) + 
  geom_polygon(color = "black", fill = "gray") +
  coord_sf(xlim=c(-88, -79.5)) 


##### Plot the number of vacancy in 2024 from RF
fmap.NVACANT.RF.2024 <- fl_base + 
  geom_polygon(data = fl_county, color = "white", fill = "gray60") +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  geom_point(data = pred.RF, aes(x = Longitude, y = Latitude, color = as.factor(predict.rf.future.2024.NVACANT)), size = 1.2, alpha = 0.5, inherit.aes = FALSE) +
  scale_color_manual(values = c("1" = "dark blue", "0" = "red"), name = "Prediction", labels = c("Less developed", "More developed")) +
  theme(legend.key.height = unit(4, "line"), legend.position = "none") +
  xlab("Longitude") +
  ylab("Latitude") +
  geom_text(data = fl_big_cities, aes(x = Longitude, y = Latitude, label = City, group = NULL, fontface = 2), size = 5, color = "orange") #+

fmap.NVACANT.RF.2024




##### Plot the number of single-family residential in 2024 from RF
fmap.NVACANT.RF.2029 <- fl_base + 
  geom_polygon(data = fl_county, color = "white", fill = "gray60") +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  geom_point(data = pred.RF, aes(x = Longitude, y = Latitude, color = as.factor(predict.rf.future.2024.NSINGLERES)), size = 1.2, alpha = 0.5, inherit.aes = FALSE) + 
  scale_color_manual(values = c("1" = "dark blue", "0" = "red"), name = "Prediction", labels = c("Less developed", "More developed")) +
  theme(legend.key.height = unit(4, "line"), legend.position = "none") +
  xlab("Longitude") +
  ylab("Latitude") +
  geom_text(data = fl_big_cities, aes(x = Longitude, y = Latitude, label = City, group = NULL, fontface = 2), size = 5, color = "orange") #+

fmap.NVACANT.RF.2029



##### Plot (the number of vacancy in 2024 = 1) & (the number of single-family residential in 2024 = 0) from RF
fmap.NV1NS0.20242024 <- fl_base + 
  geom_polygon(data = fl_county, color = "white", fill = NA) +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  theme(legend.position = "right") +
  geom_point(data = pred.RF[which(pred.RF$predict.rf.future.2024.NVACANT == 1 & pred.RF$predict.rf.future.2024.NSINGLERES == 0),], aes(x = Longitude, y = Latitude, color = as.factor(predict.rf.future.2024.NVACANT)), size = 1.2, alpha = 0.5, inherit.aes = FALSE) + 
  scale_color_manual(values = c("1" = "dark green", "0" = "dark orange"), name = "NV0 & NS1", labels = c("", "")) +
  theme(legend.key.height = unit(4, "line"), legend.position = "none") +
  xlab("Longitude") +
  ylab("Latitude") +
  geom_text(data = fl_big_cities, aes(x = Longitude, y = Latitude, label = City, group = NULL, fontface = 2), size = 5, color = "black") #+

fmap.NV1NS0.20242024




##### Plot (the number of vacancy in 2024 = 0) & (the number of single-family residential in 2024 = 1) from RF
fmap.NV0NS1.20242024 <- fl_base + 
  geom_polygon(data = fl_county, color = "white", fill = NA) +
  geom_polygon(color = "white", fill = NA) +
  theme_bw() +
  theme(legend.position = "right") +
  geom_point(data = pred.RF[which(pred.RF$predict.rf.future.2024.NVACANT == 0 & pred.RF$predict.rf.future.2024.NSINGLERES == 1),], aes(x = Longitude, y = Latitude, color = as.factor(predict.rf.future.2024.NVACANT)), size = 1.2, alpha = 0.5, inherit.aes = FALSE) + 
  scale_color_manual(values = c("1" = "dark green", "0" = "dark orange"), name = "NV0 & NS1", labels = c("", "")) +
  theme(legend.key.height = unit(4, "line"), legend.position = "none") +
  xlab("Longitude") +
  ylab("Latitude") +
  geom_text(data = fl_big_cities, aes(x = Longitude, y = Latitude, label = City, group = NULL, fontface = 2), size = 5, color = "black") #+

fmap.NV0NS1.20242024

