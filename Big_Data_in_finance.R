
#  Script in Process
#  Data not public available

 
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

#to do
#change to roc to test sample


pd_data_v2 <- read_delim("pd_data_v2.csv", 
                         ";", escape_double = FALSE, trim_ws = TRUE)


boxplot_all <- melt(pd_data_v2, id.vars = "default")%>%
  ggplot(data = , aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=default))+ facet_wrap( ~ variable, scales="free")

#on the first view lots of outlier

to.na<- c(-1e+19,-1.000000e+19,1.000000e+19)
pd_data_v2<-replace_with_na_all(pd_data_v2, condition = ~.x  %in% to.na)
save(pd_data_v2, file =  "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\BIG2\\pd_data_v2.Rda") 

pd_data_v3 <-pd_data_v2

summary(pd_data_v2)
#ask
plot_all <- melt(pd_data_v2, id.vars = "default")%>%
  ggplot(data = , aes(x=variable, y=value)) + 
  geom_point()+ facet_wrap( ~ variable, scales="free")

outlier(pd_data_v3)

summary(pd_data_v2$interest_coverage_ratio)

str(pd_data_v2)
pd_data_v2 <- pd_data_v2 %>% mutate_if(is.factor, as.numeric) %>% as.data.frame()
#corplot
pd_p <- correlate(pd_data_v2, test = TRUE)$p.value
corPlot <- correlate(pd_data_v2)$correlation %>%
  ggcorrplot(, method = "square", type = "lower", show.legend = T, 
             outline.color = "white", p.mat = pd_p, insig = "blank", tl.cex = 7,
             color = c("cyan3", "white", "firebrick2"))

save(corPlot, file ="C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\corPlot.Rda")

lm(default~., data = pd_data_v2)

pd_data_v2$default <- as.factor(pd_data_v2$default)
pd_data_v2$age_of_company<-as.factor(pd_data_v2$age_of_company)
pd_data_v2$adverse_audit_opinion <- as.factor(pd_data_v2$adverse_audit_opinion)
pd_data_v2$industry <- as.factor(pd_data_v2$industry)
pd_data_v2$payment_reminders <- as.factor(pd_data_v2$payment_reminders)



#Split in test and train data
set.seed(123)

sample<-createDataPartition(y=pd_data_v2$default,p=0.7,list=FALSE ) 
train<-pd_data_v2[sample,]
test <-  pd_data_v2[-sample,]
rm(sample)
table(train$default)
table(test$default)



# log Model
#Do parallel

detectCores()
cluster <- makeCluster(6)
registerDoParallel(cluster)
trCtrl <- trainControl(method = "repeatedcv", repeats = 1, number = 10, allowParallel = TRUE,
                       returnData = TRUE,sampling = "smote"  , savePredictions = T, classProbs = TRUE,
                       summaryFunction = twoClassSummary)



train.glm<-train
train.glm$default<-make.names(train$default)

glm.model <- train(default~., data = train.glm, method = "glm", trControl = trCtrl, family = "binomial", 
                   metric = "ROC", na.action= na.pass)

pred.glm <- predict(glm.model, newdata= test, type = "prob")[,2]
save(pred.glm, file =  "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\BIG2\\pred.glm.Rda") 
save(glm.model, file = "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\Models\\glm.model.Rda")

#ranger
weight <- if_else(train$default==1, round(sum(train$default==0)/sum(train$default==1)), 1)

grid.ranger <- expand.grid(
  mtry       = seq(3, 23, by = 5),
  node.size  = c(1, 3, 5),
  ntree      = c(500),
  maxdepth   = c(10, 20, 30),
  samp.size  = c(0.63, 0.75, 1),
  AUC       = 0
)


# Calculate all different combinations of parameters
set.seed(123)
#set index CV
cv <- createFolds(pd_data_v2$default, k=10)
#combination from the folders of cv
com<-combn(c(1:10), 9)
com <- rbind(com,10:1)
#create matrix for save cv in every round
auc.cv <- matrix(, ncol = nrow(grid.ranger), nrow =10)


number<-0
start.time <- Sys.time()

for(i in 1:nrow(grid.ranger)) {
  for ( j in 1: length(cv)){
    
    #create train and test data
    train<-pd_data_v2[unlist(cv[com[1:9,j]]), ]
    test<-as.data.frame(pd_data_v2[unlist(cv[com[10,j]]), ])  
    weight <- if_else(train$default==1,  round(sum(train$default==0)/sum(train$default==1)),1) 
    
    #run model with the data j times
    Modell.ranger <- ranger(
      formula         = default ~ ., 
      data            = train,
      num.trees       = grid.ranger$ntree[i],
      mtry            = grid.ranger$mtry[i],
      min.node.size   = grid.ranger$node.size[i], 
      max.depth       = grid.ranger$maxdepth[i],
      sample.fraction = grid.ranger$samp.size[i],
      case.weights    = weight,
      seed            = 123, 
      classification  = T, 
      probability     = T
    )
    
    #predict the auc for the combination of parameters i
    pred<-stats::predict(Modell.ranger, test, type="response")
    auc.cv[j,i]<- roc(response = test$default, predictor = pred$predictions[,2])$auc
  } 
  
  # Calculation AUC
  grid.ranger$AUC[i] <- mean(auc.cv[i], na.rm = T) 
  number<-number+1
  print(number)
}
end.time <- Sys.time()
time.ranger<-end.time-start.time


# Which parameter combination has the lowest AUC?
grid.ranger[which.max(grid.ranger$AUC), ]
max <- which.max(grid.ranger$AUC)

save(grid.ranger, file = "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\Models\\grid.ranger.Rda")

# Calculate Finales Model 
final.ranger <- ranger(
  formula         = default ~ ., 
  data            = train, 
  num.trees       = grid.ranger$ntree[max],
  mtry            = grid.ranger$mtry[max],
  min.node.size   = grid.ranger$node.size[max], 
  max.depth       = grid.ranger$maxdepth[max],
  sample.fraction = grid.ranger$samp.size[max],
  seed            = 123, 
  importance      = 'impurity'
)

save(final.ranger, file = "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\Models\\finale.ranger.Rda")







#xgb


step_num2factor(Sex...Marital.Status,
                Payment.Status.of.Previous.Credit,
                Purpose,
                Occupation,
                Telephone,
                Foreign.Worker)

# Create recipe
rec <- recipe(default ~ ., data = pd_data_v2) %>%
  step_mutate(default = as.numeric(as.character(default))) %>% 
  step_log(all_numeric(), offset = 1) %>%
  step_dummy(all_nominal())



# Train the recipe on the training set.
prep_rec <- prep(rec, training = train)

# Bake the data (i.e. apply the recipe and get the final datasets)
mod_train <- bake(prep_rec, new_data = train)
mod_test <- bake(prep_rec, new_data = test)


xgb.matrix.train<- xgb.DMatrix(as.matrix(mod_train %>% select(-default)), 
                               label = mod_train$default)
xgb.matrix.test <- xgb.DMatrix(as.matrix(mod_test %>% select(-default)), 
                               label = mod_test$default)






grid.xgb<-expand.grid(eta=c(0.1,0.05, 0.01),
                      max_depth=c(2,5,7,10),
                      subsample=0.75,
                      colsample_bytree= c(0.5,0.7,1),
                      gamma = 0,
                      min_child_weight = 2,
                      ntree=c(200, 400))


number<-0
AUC<-data.frame(grid.xgb)
AUC$auc<-0
start.time <- Sys.time()
?xgb.cv
for ( i in 1:nrow(grid.xgb)){
  cv.xgb <-xgb.cv( data = xgb.matrix.train , seed=123, 
                   scale_pos_weight=93,
                   objective="binary:logistic",
                   booster="gbtree",
                   eval_metric= "auc",
                   verbose=T,
                   nfold = 10,
                   nrounds = grid.xgb$ntree[i],
                   eta=grid.xgb$eta[i],
                   max_depth=grid.xgb$max_depth[i],
                   subsample=grid.xgb$subsample[i],
                   colsample_bytree= grid.xgb$colsample_bytree[i], 
                   min_child_weight = grid.xgb$min_child_weight[i], 
                   gamma = grid.xgb$gamma[i]
                   prediction = T, 
                   tree_method= "hist", 
                   early_stopping_rounds=5)
  AUC$auc[i] <- cv.xgb[["evaluation_log"]][["test_auc_mean"]][cv.xgb$niter]
  
  number<-number+1
  print(number)
}
end.time <- Sys.time()
time.xgb <- end.time-start.time

AUC[which.max(AUC$auc),]
max<-which.max(AUC$auc)
save(AUC, file = "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\Models\\tune.xgb.auc.Rda")

model.xgb <- xgb.train( data = xgb.matrix.train , seed=123, 
                        scale_pos_weight=93,
                        objective="binary:logistic",
                        booster="gbtree",
                        eval_metric= "auc",
                        verbose=F,
                        nrounds = grid.xgb$ntree[max],
                        eta=grid.xgb$eta[max],
                        max_depth=grid.xgb$max_depth[max],
                        subsample=grid.xgb$subsample[max],
                        colsample_bytree= grid.xgb$colsample_bytree[max],
                        tree_method= "hist", 
                        early_stopping_rounds=5)

save(model.xgb, file = "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\Models\\xgb.model.Rda")

model.xgb <- xgb.train( data = xgb.matrix.train , seed=123, 
                        scale_pos_weight=93,
                        objective="binary:logistic",
                        booster="gbtree",
                        eval_metric= "auc",
                        verbose=F,
                        nrounds = 400,
                        eta=0.05,
                        max_depth=2,
                        subsample=0.75,
                        colsample_bytree=0.7)


pred.xgb <- predict(model.xgb, newdata = xgb.matrix, type="prob")
save(pred.xgb, file = "C:\\Users\\Paul\\Documents\\Uni\\Bergen\\Big Data\\BIG\\Big2\\pred.xgb.Rda")
roc(response=train$default, predictor = pred.xgb)$auc
roc.xgb<-roc(response=train$default, predictor = pred.xgb)
ggroc(roc.xgb, alpha = 0.5, colour = "red", linetype = 1, size = 1) + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color="black", linetype="dashed")+ xlab("FPR") + ylab("TPR") 


ggplot(melt(AUC, id.vars = "auc"), aes(x=value, y= auc, color=variable)) + geom_point()




melt.data<-data.frame(roc.xgb$sensitivities, roc.xgb$specificities, roc.xgb$thresholds)%>%
  melt(, id.vars = "roc.xgb.thresholds")

ggplot(melt.data,aes(x=roc.xgb.thresholds,y=value, color=variable)) + 
  geom_line()+
  ylab("TNR / TPR")+
  xlab("thresholds")+
  theme(legend.title = element_text(colour="chocolate", size=16, face="bold"))+
  scale_color_discrete(name = "Variable", labels = c("TNR", "TPR"))

plot(density(pred.xgb))
lines(density(pred.glm), col = "red")
lines(density(final.ranger$predictions-1),col = "blue" )
