#This script is part of a project of mine at the norwegian school of economics. 
#It is divided into data preparation, GLM, Random Forest via Ranger with cross validation, 
#Gradient Boosting via xgb also with cross validation and finally the used plots. 








# Load the necessary packages
library(tidyverse)
library(ggplot2)
library(xgboost)
library(pROC)
library(caret)
library(doParallel)
library(ranger)
library(car)
library(reshape2)
library(naniar)
library(outliers)
library(ggcorrplot)
library(lsr)
library(recipes)
library(missRanger)
library(scales)
library(viridis)
library(hrbrthemes)



###### DATA
# Load the data
pd_data_v2 <- read_delim("pd_data_v2.csv", 
                         ",", escape_double = FALSE, trim_ws = TRUE)


# At the first view lots of high outliers these don't seem natural
summary(pd_data_v2)

# Change outliers non natural outliers to NA
to.na<- c(-1e+19,-1.000000e+19,1.000000e+19)
pd_data_v2<-replace_with_na_all(pd_data_v2, condition = ~.x  %in% to.na)

str(pd_data_v2)
# Change that have to be factor variable to factor
col <- c("default", "age_of_company",
         "adverse_audit_opinion", 
         "industry", "payment_reminders")
pd_data_v2[col] <- lapply(pd_data_v2[col], as.factor)

# How is the ratio between default and non default. Is the Data imbalenced?
# The date is high imbalaned
data.frame(Default=round(sum(pd_data_v2$default==1)/length(pd_data_v2$default), 3), Non_default=
             round(sum(pd_data_v2$default==0)/length(pd_data_v2$default), 3))


#Definintion of train and test dataset
sample<-createDataPartition(y = pd_data_v2$default,
                            p = 0.7,
                            list = FALSE) 

train <- pd_data_v2[sample,]
test  <-  pd_data_v2[-sample,]
table(train$default)


# Impute from NAs for the modelling with glm and ranger
train_miss <- missRanger(train,
                         seed = 123,
                         maxiter = 4, 
                         num.trees=200)

test_miss <- missRanger(test, 
                        seed = 123,
                        maxiter = 4,
                        num.trees=200 )


###### GLM

# The GLM work with names, not with factors
train.glm <- train_miss
train.glm$default <- make.names(train_miss$default)

# Choice how much cores du we use
detectCores()
makeCluster(5) %>%
  registerDoParallel() 

#Train control parameters are fixed. 10 times cross validation
trCtrl <- trainControl(method = "repeatedcv", 
                       repeats = 1,
                       number = 10, 
                       allowParallel = TRUE,
                       returnData = TRUE ,
                       savePredictions = T,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary, 
                       sampling = "up")




# GlM model with trCTrl as input and AUC metric
glm.model <- train(default~., 
                   data = train.glm,
                   method = "glm",
                   trControl = trCtrl, 
                   family = "binomial", 
                   metric = "ROC",
                   na.action= na.pass)

# Prediction from the test dataset with the glm model
pred.glm <- predict(glm.model,
                    newdata= test_miss, 
                    type = "prob")[,2]
#Calculate the ROC
roc.glm<-roc(response = test_miss$default, predictor = pred.glm)
#AUC
roc.glm$auc
#0.8853




# Random Forest over Ranger

# Hyperparmeter
grid.ranger <- expand.grid(
  mtry       = seq(5, 23, by = 6),
  node.size  = c(1, 3, 5),
  ntree      = c(500),
  maxdepth   = c(10, 20, 30),
  samp.size  = c(0.63, 0.75, 1),
  AUC       = 0
)

# Calculate all different combinations of parameters
set.seed(123)
#set index CV
cv <- createFolds(train_miss$default, k=)
#combination from the folders of cv
com <- combn(c(1:5), 4) %>% rbind(5:1)

#create matrix for save cv in every round
auc.cv <- matrix(, ncol = nrow(grid.ranger), nrow =5)


number<-0
start.time <- Sys.time()
for(i in 1:nrow(grid.ranger)) {
  for ( j in 1: length(cv)){
    
    #create train and test data
    train.cv<-train_miss[unlist(cv[com[1:4,j]]), ]
    test.cv<-as.data.frame(train_miss[unlist(cv[com[5,j]]), ])  
    #Create weight from the trainings dataset
    weight <- if_else(train.cv$default==1,
                      round(sum(train.cv$default==0)/sum(train.cv$default==1)),1) 
    
    
    #run model with the data j times
    Modell.ranger <- ranger(
      formula         = default ~ ., 
      data            = train.cv,
      num.trees       = grid.ranger$ntree[i],
      mtry            = grid.ranger$mtry[i],
      min.node.size   = grid.ranger$node.size[i], 
      max.depth       = grid.ranger$maxdepth[i],
      sample.fraction = grid.ranger$samp.size[i],
      weight          = weight,
      seed            = 123, 
      classification  = T, 
      probability     = T
    )
    
    #predict the auc for the combination of parameters i
    pred<-stats::predict(Modell.ranger, test.cv, type="response")
    auc.cv[j,i]<- roc(response = test.cv$default,
                      predictor = pred$predictions[,2])$auc
  } 
  
  # Calculation AUC
  grid.ranger$AUC[i] <- mean(auc.cv[i], na.rm = T) 
  number<-number+1
  print(number)
}
end.time <- Sys.time()
time.ranger<-end.time-start.time


# Which parameter combination has the highest l AUC?
grid.ranger[which.max(grid.ranger$AUC), ]
max <- which.max(grid.ranger$AUC)


# Calculate Finales Model 
final.ranger <- ranger(
  formula         = default ~ ., 
  data            = train_miss, 
  num.trees       = 500,
  mtry            = head(grid.ranger[order(grid.ranger$AUC,decreasing = T ),],3)$mtry[3],
  min.node.size   = head(grid.ranger[order(grid.ranger$AUC,decreasing = T ),],3)$node.size[3],
  max.depth       = head(grid.ranger[order(grid.ranger$AUC,decreasing = T ),],3)$maxdepth[3],
  sample.fraction = head(grid.ranger[order(grid.ranger$AUC,decreasing = T ),],3)$samp.size[3],
  seed            = 123, 
  case.weights    = weight,
  probability     = T,
  importance      = 'impurity'
)

# Predict the test data with the final model
pred.ranger<-predict(final.ranger, 
                     test_miss,
                     type = "response")

# Calculate the ROC
roc.ranger.test<-roc(response=test_miss$default, 
                     predictor = pred.ranger$predictions[,2])
#AUC
roc.ranger.test$auc
#0.9087


#XGB
# Weights based on the train dataset to handel imbalanced data in xgb
weight <- if_else(train$default==1,
                  round(sum(train$default==0)/sum(train$default==1)),1)


# Create recipe
rec <- recipe(default ~ ., data = pd_data_v2) %>%
  step_mutate(default = as.numeric(as.character(default))) %>% 
  step_num2factor(age_of_company,
                  adverse_audit_opinion,
                  payment_reminders,
                  industry)%>%
  step_log(all_numeric(), offset = 1) %>%
  step_dummy(all_nominal())

# Change train and test data  back to be numeric because prep need numeric input
train[col] <- lapply(train[col], as.character)
train[col] <- lapply(train[col], as.numeric)
test[col] <- lapply(test[col], as.character)
test[col] <- lapply(test[col], as.numeric)



# Train the recipe on the training set.
prep_rec <- prep(rec, training = train)
prep_rec_test <- prep(rec, training = test)

# Bake the data
mod_train <- bake(prep_rec, new_data = train)
mod_test <- bake(prep_rec_test, new_data = test)
mod_train$default <- train$default
mod_test$default <- test$default

# Create the XGB Matrix and fic the label
xgb.matrix.train <- xgb.DMatrix(as.matrix(mod_train %>% select(-default)), 
                                label = mod_train$default)
xgb.matrix.test <- xgb.DMatrix(as.matrix(mod_test %>% select(-default)), 
                               label = mod_test$default)

# Hyperparmeter
grid.xgb <- expand.grid(eta = c(0.1, 0.05, 0.01),
                        max_depth = c(2, 5, 10, 12),
                        subsample = c(0.75, 1),
                        colsample_bytree = c(0.5, 0.7, 1),
                        gamma = c( 1, 2),
                        min_child_weight = c(3, 5),
                        ntree = c(200, 400),
                        auc = 0)



# Start the Tuning of the xgb model
# 
number<-0
start.time <- Sys.time()
for (i in 1:nrow(grid.xgb)){
  cv.xgbi <-xgb.cv(data = xgb.matrix.train , seed=123, 
                   objective = "binary:logistic",
                   booster = "gbtree",
                   eval_metric = "auc",
                   verbose = F,
                   nfold = 10,
                   nrounds = grid.xgb$ntree[i],
                   eta = grid.xgb$eta[i],
                   max_depth = grid.xgb$max_depth[i],
                   subsample = grid.xgb$subsample[i],
                   colsample_bytree = grid.xgb$colsample_bytree[i], 
                   min_child_weight = grid.xgb$min_child_weight[i], 
                   gamma = grid.xgb$gamma[i],
                   weight = weight,
                   prediction = T, 
                   tree_method = "hist",
                   early_stopping_rounds=5)
  
  grid.xgb$auc[i] <- cv.xgb[["evaluation_log"]][["test_auc_mean"]][cv.xgb$niter]
  number+1
  print(number)
}
end.time <- Sys.time()
time.xgb <- end.time-start.time

grid.xgb[which.max(grid.xgb$auc),]
max<-which.max(data_xgb$auc)

model.xgb <- xgb.train( data = xgb.matrix.train , seed=123, 
                        objective ="binary:logistic",
                        booster ="gbtree",
                        eval_metric = "auc",
                        verbose = F,
                        nrounds = grid.xgb$ntree[max],
                        eta = grid.xgb$eta[max],
                        max_depth = grid.xgb$max_depth[max],
                        subsample = grid.xgb$subsample[max],
                        colsample_bytree = grid.xgb$colsample_bytree[max],
                        min_child_weight = grid.xgb$min_child_weight[max], 
                        gamma = grid.xgb$gamma[max],
                        weight = weight,
                        tree_method= "hist")




pred.xgb <- predict(model.xgb, 
                    xgb.matrix.test,
                    type="prob")
roc.xgb <- roc(response=test$default, 
               predictor = pred.xgb)




c(roc.glm$auc, roc.ranger.test$auc, roc.xgb$auc)
#0.8852930 0.9086566 0.9313508


#calculate best threshold as table
threshold <- rbind(  as.vector(coords(roc.glm, "best", "threshold", transpose = FALSE)), 
                     as.vector(coords(roc.ranger.test, "best", "threshold", transpose = FALSE)), 
                     as.vector(coords(roc.xgb, "best", "threshold", transpose = FALSE)))
row.names(threshold) <- c("GLM", "Ranger", "XGB")



#Plots

com_mod <- rbind(data.frame(sensitivities=roc.glm$sensitivities,
                            specificities=roc.glm$specificities,
                            thresholds=roc.glm$thresholds, 
                            label="glm"),
                 data.frame(sensitivities=roc.ranger.test$sensitivities,
                            specificities=roc.ranger.test$specificities, 
                            thresholds=roc.ranger.test$thresholds,
                            label="ranger"),
                 data.frame(sensitivities=roc.xgb$sensitivities,
                            specificities=roc.xgb$specificities,
                            thresholds=roc.xgb$thresholds, 
                            label="xgb"))%>%
  
  ggplot() + 
  geom_line(aes(x=thresholds,y=sensitivities, colour="TPR"))+
  geom_line(aes(x=thresholds,y=specificities, colour="TNR"))+
  ylab("TNR / TPR")+
  xlab("Thresholds")+
  facet_wrap( ~ label, 
              strip.position = "bottom", 
              scales = "free_x")+ scale_colour_manual(name="",
                                                      values=c(TPR="cornflowerblue", TNR="firebrick1"))+
  theme( axis.title.y = element_blank())


#Boxplot prob
melt_all_pred<-data.frame(default=test$default, 
                          glm=pred.glm,
                          ranger=pred.ranger$predictions[,2] ,
                          xgb=pred.xgb)%>%
  melt(, id.vars = "default")%>%
  
  ggplot( aes(x=variable, 
              y=value, 
              fill=default)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE,
                     alpha=0.6, 
                     option="A") +
  theme_ipsum() +
  facet_wrap( ~ default, 
              strip.position = "bottom",
              scales = "free_x", 
              labeller = as_labeller(c("0"="Non-Default ", 
                                       "1"="Default") )) +
  geom_jitter(color="black", 
              size=0.1, 
              alpha=0.08) +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Compare prob. Models") +
  xlab("")+ 
  ylab("Prob.Default")


#ROC
roc_ranger <- roc(test_miss$default,pred.ranger$predictions[,2])
roc_glm <- roc(test_miss$default, pred.glm)
roc_xgb <- roc(test$default, pred.xgb)

roc_all <- ggroc(list(GLM = roc_glm,
                      Ranger = roc_ranger, 
                      xgBoost = roc_xgb), 
                 size = 1.5) + 
  geom_segment(aes(x = 1, xend = 0,
                   y = 0, yend = 1),
               color="black",  
               linetype="dashed") +
  scale_fill_discrete(name = "",
                      labels= c("GLM", "Ranger", "XGB")  )

# Plot differences in prediction between Ranger and XGB
com_diff <- data.frame(pred.ranger=pred.ranger$predictions[,2],
                       xgb=pred.xgb,
                       diff.ranger.minus.xgb=pred.ranger$predictions[,2]- pred.xgb,
                       default=test$default)%>%
  
  ggplot(aes(x=pred.ranger, y=diff.ranger.minus.xgb))+
  geom_point()+
  geom_abline(slope = 0, 
              intercept = 0, 
              col ="red")+
  facet_wrap( ~ default,
              strip.position = "bottom", 
              scales = "free_x" ,  
              labeller = as_labeller(c("0"="Non-Default ", "1"="Default") ))+
  xlab("Prediction Ranger")+ 
  ylab("Prediction.ranger - Prediction.xgb")


