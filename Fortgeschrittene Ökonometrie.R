# The analysis is based on the data set of https://github.com/amanthedorkknight/fifa18-all-player-statistics
# The aim of the study was to find a model that predicts the weekly income of football players.

# The code listed here is part of a group project.
# The data preparation and overview were developed together and were therefore left in the code. 
# The parts not written by me included Linear Regression, Principal Components Regression and Partial Least Squares 
# and were removed from the code. 



### Packages

library(lubridate)
library(measurements)
library(readxl)
library(plyr)
library(dplyr)
library(caret)
library(ggplot2)
library(ggcorrplot)
library(lsr)
library(caret)
library(doParallel)
library(MLmetrics)
library(gridExtra)
library(car)
library(ranger)
library(vip)
library(gbm)
library(scales)


### Data preparation

load("FIFA19_train.Rdat")
summary(train)

# Delete from the columns X1, ID, NAme, Photo, Flag, Club Logo and Real Face. 
train <- subset(train,  select = -c(X1, ID, Name, Photo, Flag, get('Club Logo'), get('Real Face')))

# Position Skills removed because too many NAs 
train <- subset(train, select = -c(LS, ST, RS, LW, LF, CF, RF, RW, LAM, 
                                   CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, 
                                   CDM, RDM, RWB, LB, LCB, CB, RCB, RB))

# Delete players without personal attributes 
train[is.na(train$Weight), ]
train <- train[!is.na(train$Weight), ]

# Delete players without club and therefore without salary (1.279 percent of observations) 
sum(is.na(train$Club))/length(train$Club)
train <- train[!is.na(train$Club), ]

# Some entries in Body Type incorrect (problem with sofifa)
# Information about height and weight of players available, therefore body type removed
train <- subset(train, select = -`Body Type`)

# Lending club conducted salary negotiations at contract conclusion with the players.
train$Club[!is.na(train$`Loaned From`)] <- train$`Loaned From`[!is.na(train$`Loaned From`)]

# Dummy variable (player lent or not)
train$`Loaned From`[!is.na(train$`Loaned From`)] <- 1
train$`Loaned From`[is.na(train$`Loaned From`)] <- 0
names(train)[names(train) == "Loaned From"] <- "Loaned"
## Welche Klasse muss diese Variablen haben?

# Value, Wage and Release Clause edit
train$Value <- sub("\u20AC", "", train$Value)
train$Value <- sub("K", "000", train$Value)
train$Value[grep("M", train$Value)] <- as.numeric(sub("M","",train$Value[grep("M", train$Value)] ))*10^6
train$Value<-as.numeric(train$Value)

train$Wage <- sub("\u20AC", "", train$Wage)
train$Wage <- as.numeric(sub("K", "000", train$Wage))

train$`Release Clause` <- sub("\u20AC", "", train$`Release Clause`)
train$`Release Clause` <- sub("K","000", train$`Release Clause`)
train$`Release Clause`[grep("M", train$`Release Clause`)] <- as.numeric(sub("M","",train$`Release Clause`[grep("M", train$`Release Clause`)]))*10^6
train$`Release Clause` <- as.numeric(train$`Release Clause`)

#If no sales clause is stipulated, the price results from negotiations.
#In the case of loaned players, no contractually agreed replacements are known.
#In both cases the values are set to zero.
train$`Release Clause`[is.na(train$`Release Clause`)] <- 0

# Contract Valid Until does not indicate the end of the contract for lent players, but the end of the loan.
# Lent players had their salary negotiated when they signed the actual contract with the lending club.
# There is no information available about the beginning and end of the contract of lent players.
# Either the lent players or the start and end of the contract would have to be removed.
# Since the lending status is likely to have a greater impact on
# will have the weekly salary, the Joined and Contract Valid Until variables will be removed.

Translated with www.DeepL.com/Translator
train <- subset(train, select = -c(Joined, `Contract Valid Until`))

# Convert height from inches to centimeters
Ft <- as.numeric(substr(train$Height, start = 1, stop = 1))
Ft <- conv_unit(Ft, "ft", "cm")
Inches <- as.numeric(substr(train$Height, start = 3, stop = 4))
Inches <- conv_unit(Inches, "inch", "cm")
train$Height <- Ft + Inches

# Convert Weight from Pfund to kg
train$Weight <- as.numeric(sub("lbs", "", train$Weight))
train$Weight <- round(conv_unit(train$Weight, "lbs", "kg"), 1)

# To compress information and use the origin in tree based models the countries are assigned to regions.
# Country to region allocation from the world bank
Laender_Region_zuordnung <- read_excel("~Country_Code.xls", 
                                       skip = 5)
Laender_Region_zuordnung <- Laender_Region_zuordnung[,-c(1,2,4,5,7,8,9)]
names(Laender_Region_zuordnung)[1] <- "Nationality"
train$Nationality[which(train$Nationality=="England" |train$Nationality== "Scotland"|train$Nationality=="Wales"|train$Nationality== "Northern Ireland")]<-"United Kingdom"
train$Nationality[which(train$Nationality=="Republic of Ireland")]<-"Ireland"
train$Nationality[which(train$Nationality=="Korea Republic")]<-"Korea, Rep."
train$Nationality[which(train$Nationality=="Korea DPR")]<-"Korea, Dem. People's Rep."
train$Nationality[which(train$Nationality=="Congo")]<-"Congo, Rep."
train$Nationality[which(train$Nationality=="DR Congo")]<-"Congo, Dem. Rep."
train$Nationality[which(train$Nationality=="Syria")]<-"Syrian Arab Republic"
train$Nationality[which(train$Nationality=="Iran")]<-"Iran, Islamic Rep."
train$Nationality[which(train$Nationality=="Curacao")]<-"Curaçao"
train$Nationality[which(train$Nationality=="Bosnia Herzegovina")]<-"Bosnia and Herzegovina"
train$Nationality[which(train$Nationality=="Venezuela")]<-"Venezuela, RB"
train$Nationality[which(train$Nationality=="Slovakia")]<-"Slovak Republic"
train$Nationality[which(train$Nationality=="Egypt")]<-"Egypt, Arab Rep."
train$Nationality[which(train$Nationality=="FYR Macedonia")]<-"North Macedonia"
train$Nationality[which(train$Nationality=="Cape Verde")]<-"Cabo Verde"
train$Nationality[which(train$Nationality=="Ivory Coast")]<-"Côte d'Ivoire"
train$Nationality[which(train$Nationality=="China PR")]<-"China"
train$Nationality[which(train$Nationality=="Russia")]<-"Russian Federation"
train$Nationality[which(train$Nationality=="Gambia")]<-"Gambia, The"
train <- join(x = train , y = Laender_Region_zuordnung , by = "Nationality", )
names(train)[ncol(train)]<-"Region"
train$Region[is.na(train$Region)]<-"missing"

# Remove Nationality, otherwise Prediction later not possible
train <- subset(train, select = -Nationality)

# Dummyvariablen 
str(train)
train = train %>% mutate_if(is.character, as.factor)

#Names without spaces
colnames(train)<-make.names(colnames(train), unique=TRUE)
summary(train)

# Save Data
save(train, file = "FIFA_bereinigt.Rda")

# Test- und Traindataset 
set.seed(3456)
train.Index <- createDataPartition(train$Wage, p = 0.7,
                                   list = FALSE, times = 1)
head(train.Index)
FIFA_train <- train[train.Index, ]
FIFA_test <- train[- train.Index, ]

save(FIFA_train, file = "FIFA.train.Rda")
save(FIFA_test, file = "FIFA.test.Rda")

# Boxplot Wage

# Boxplot
ggplot(train, aes(x = Wage)) + geom_boxplot(aes(y = Wage), alpha = 0.8,fill = "cyan3", 
                                            size = 0.4) + coord_flip() + xlab("")


# Calculate the correlation of the numeric variables.
# Non-significant values are hidden.
FIFA_cor <- correlate(train)$correlation
p.FIFA <- correlate(train, test = TRUE)$p.value

ggcorrplot(FIFA_cor, method = "square", type = "lower", show.legend = T, 
           outline.color = "white", p.mat = p.FIFA, insig = "blank", tl.cex = 7,
           color = c("cyan3", "white", "firebrick2"))





### Random Forest and Boosting

### Ranger

# Set tuning parameters
grid.ranger <- expand.grid(
  mtry       = seq(20, 52, by = 4),
  node.size  = c(3, 4, 5),
  ntree      = c(200, 400, 600, 800, 1000),
  maxdepth   = c(10, 20, 30),
  samp.size  = c(0.63, 0.75, 1),
  RMSE       = 0,
  rss        = 0
)

# Calculate all different combinations of parameters
for(i in 1:nrow(grid.ranger)) {
  Modell.ranger <- ranger(
    formula         = Wage ~ ., 
    data            = FIFA_train, 
    num.trees       = grid.ranger$ntree[i],
    mtry            = grid.ranger$mtry[i],
    min.node.size   = grid.ranger$node.size[i], 
    max.depth       = grid.ranger$maxdepth[i],
    sample.fraction = grid.ranger$samp.size[i],
    seed            = 123
  )
  
  # Calculation RMSE
  grid.ranger$RMSE[i] <- sqrt(Modell.ranger$prediction.error)
  grid.ranger$rss[i]  <- Modell.ranger$r.squared
} 

# Which parameter combination has the lowest RMSE?
grid.ranger[which.min(grid.ranger$RMSE), ]
min <- which.min(grid.ranger$RMSE)

# Calculate Finales Model 
final.ranger <- ranger(
  formula         = Wage ~ ., 
  data            = FIFA_train, 
  num.trees       = grid.ranger$ntree[min],
  mtry            = grid.ranger$mtry[min],
  min.node.size   = grid.ranger$node.size[min], 
  max.depth       = grid.ranger$maxdepth[min],
  sample.fraction = grid.ranger$samp.size[min],
  seed            = 123, 
  importance      = 'impurity'
)

final_pred <- predict(final.ranger, FIFA_test)
RMSE(final_pred$predictions, FIFA_test$Wage)
# [1] 9399.45

# Calculation of the significance of variables in the Ranger Regression
werte <- as.vector(head(final.ranger$variable.importance, 15))

# Preparation to plot the variable importance
werte <- rescale(werte, to = c(1, 100)) 
names <- as.vector((names(head(final.ranger$variable.importance,15))))
ImVar <- cbind(names, werte)
ImVar <- as.data.frame(ImVar)
ImVar$werte <- as.numeric(levels(ImVar$werte))[ImVar$werte]

# Plot Variable Importance
ggplot(ImVar, aes(x=reorder(names,werte), y=werte,fill=werte ))+
  geom_bar(stat="identity",
           position="dodge",
           fill = "cyan3")+
  coord_flip()+
  ylab("")+
  xlab("")+
  ggtitle("")+
  guides(fill=F)


### GBM

# Several cores are used simultaneously for faster calculation
cores <- makeCluster(5)
registerDoParallel(cores = cores)

# CV
gbm.Control <- trainControl(method = 'cv',
                            number = 5,
                            search = 'grid', 
                            allowParallel = TRUE)

#  GBM Model
set.seed(123)
Modell.gbm <- train(Wage ~ .,
                    data = FIFA_train,
                    method = "gbm",
                    distribution = "gaussian",
                    trControl = Allcontrol,
                    verbose = F)

#Forecast
tree <- Modell.gbm$finalModel$n.trees

gbm.pred <- predict(Modell.gbm, FIFA_test, n.trees = tree)
RMSE(gbm.pred, FIFA_test$Wage)
# 8010.308


### GBM Tuning

# Set Tuning parameter
grid.gbm <- expand.grid(n.trees = c(100, 300, 500),
                        interaction.depth = c(4, 5, 6),
                        shrinkage = c(0.01, 0.05, 0.1),
                        n.minobsinnode = c(3, 4, 5))   

# Calculate the different parameter combinations
set.seed(123)
Modell.gbm.tune <- train(Wage ~ .,
                         data = FIFA_train,
                         method = "gbm",
                         distribution = "gaussian",
                         trControl = gbm.Control,
                         verbose = F,
                         tuneGrid = grid.gbm)

# Model with the lowest CV-RMSE
Modell.gbm.tune$finalModel$tuneValue

# Forecast
tree <- Modell.gbm.tune$finalModel$n.trees

gbm.pred.tune <- predict(Modell.gbm.tune, FIFA_test, n.trees = tree)
RMSE(gbm.pred.tune, FIFA_test$Wage)
# [1] 5754.339

# Plot Variable Importance from GBM Tune
vip(Modell.gbm.tune,color = "cyan3" , fill= "cyan3", num_features = 15 )