
### CYO Andres Gomez

# Download data to be used, only white wines, and create a DF (actually a spec_tbl_df type of file)
# Download data to be used and create a DF (actually a spec_tbl_df type of file)
if(!require(readr)) install.packages("readr",  repos = "http://cran.us.r-project.org")

if(!file.exists("winequality-white.csv")) 
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", "winequality-white.csv")

wines <- read_delim("winequality-white.csv", 
                    delim = ";", 
                    locale = locale(decimal_mark = ".", 
                                    grouping_mark = ","), 
                    col_names = TRUE)

# Set column names
cnames <- c("fixed_acidity", "volatile_acidity", "citric_acid","residual_sugar", "chlorides",
            "free_sulfur_dioxide","total_sulfur_dioxide", "density", "pH","sulphates",
            "alcohol", "quality")
# Rename columns to make it friendlier for R (at least for me)
colnames(wines) <- cnames
# Quality is numeric,
# let's create a variable "rating" that will be quality as factor with convenient format
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

wines <- mutate(wines, 
                rating = as.factor(quality))
levels(wines$rating) <- paste0("R_", levels(wines$rating))

#summary stats
summary(wines)

# Distribution of quality values
ratings_prop <- wines %>% 
  group_by(rating) %>% 
  summarise(n = n()) %>% 
  mutate(prop = n/sum(n))

if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

ggplot(ratings_prop) + geom_col(aes(rating, y = prop), fill="cornflowerblue") + 
  geom_text(aes(x = rating,y = prop + 0.05, label = round(prop, 3)))+
  labs(title = "Distribution of quality (rating) for white wines",
       caption = "data: entire white whines dataset")

# Box plots
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
df.m <- melt(wines[,-c(12)], id.var = "rating")
p <- ggplot(data = df.m, aes(x=variable, y=value)) 
p <- p + geom_boxplot()
p <- p + facet_wrap( ~ variable, scales="free")
p <- p + ggtitle("Box plots")
p <- p + guides(fill=guide_legend(title="Legend_Title"))
p 
# Box plots by wine rating
p <- ggplot(data = df.m, aes(x=variable, y=value)) 
p <- p + geom_boxplot(aes(fill = rating))
p <- p + facet_wrap( ~ variable, scales="free")
p <- p + ggtitle("Box plots")
p <- p + guides(fill=guide_legend(title="Legend_Title"))
p 

#Correlation matrix
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
res <- cor(as.matrix(wines[,1:12]))
res <- round(res,2)
corrplot(res, method = "shade", type = "lower", addCoef.col = "grey" )

#prepare data for modeling
wines_m <- wines %>% 
  mutate(recom = factor(case_when(
    rating %in% c("R_3", "R_4","R_5") ~ "avoid",
    rating %in% c("R_6", "R_7","R_8","R_9") ~ "buy",
    TRUE ~ "other"
  )))

#create training and testing sets
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# Test set will be 10% of the entire dataset
set.seed(229, sample.kind = "Rounding")
#set.seed(229) if you are using an R version earlier than 3.6
indxTrain <- createDataPartition(y = wines_m$recom, 
                                 times = 1, 
                                 p = 0.9, 
                                 list = FALSE)
# Train and test sets for wine type
training <- wines_m[indxTrain,]
testing  <- wines_m[-indxTrain,]

#check that they are consistent with aggregated data
summary(training)
summary(testing)

## KNN
#pre knn
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ISLR)) install.packages("ISLR", repos = "http://cran.us.r-project.org")
trainX <- training[,names(training) != "recom"]
trainX_Rel <-trainX[,-c(12,13)]
preProcValues <- preProcess(x = trainX_Rel,method = c("center", "scale"))
preProcValues

#training and training control 
set.seed(229, sample.kind = "Rounding")
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(recom ~ ., data = training[,-c(12,13)], method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

#Output of kNN fit
knnFit

#Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
plot(knnFit)

knnPredict <- predict(knnFit,newdata = testing[,-c(12,13)] )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, testing$recom )

mean(knnPredict == testing$recom)

if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")

knnPredict <- predict(knnFit,newdata = testing , type="prob")
knnROC <- roc(testing$recom,knnPredict[,"avoid"])
knnROC

plot(knnROC, type="S", print.thres= 0.5)

##Random Forest
#the required packages are already installed
set.seed(229, sample.kind = "Rounding")
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
# Random forrest
rfFit <- train(recom~., data=training[,-c(12,13)],method="rf",trControl= ctrl,preProcess=c("center","scale"),tuneLength = 20)

rfFit

rfPredict <- predict(rfFit,newdata = testing[,-c(12,13)] )
confusionMatrix(rfPredict, testing$recom )

rfPredict <- predict(rfFit,newdata = testing[,-c(12,13)] , type="prob")
rfROC <- roc(testing$recom,rfPredict[,"avoid"])
rfROC

plot(rfROC, type="S", print.thres= 0.5)
