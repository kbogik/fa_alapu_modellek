#betöltése:
  {r, message=FALSE}
library(tm)
library(dplyr)
library(ggplot2)
library(readxl)
library(readr)
library(caret)
library(dplyr)
library(pROC)
#install.packages("xgboost")
library(xgboost)
library(randomForest)
library(gbm)
library(caretEnsemble)
library(rpart.plot)
library(plotmo)



#https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

#Ensemble####


#Adat beolvasása:
  
wineraw<- read_csv("winequality-red.csv",show_col_types = FALSE)

#Nomralizálás:
  
norm <- function(x) (x-min(x))/(max(x)-min(x))
wine<- sapply(wineraw[ ,-12], norm)
wine <- cbind(wine,quality=wineraw[ ,12])
head(wine,5)


#teljesül a 10p<n hüvelykujj szabály


summary(wine$quality)

#nincsenek NA-k

#Láthatóan a borminőség 3 és 8 közé esik.

#Vagy több kategóriát becslünk (3-8) vagy azt mondjuk, hogy 7 fölött jó a vörösbor minősége (bináris ötlet: kaggle-ről)


colnames(wine) <- c("fixedacidity", "volatileacidity","citricacid","residualsugar","chlorides","freesd","totalsd","density","pH","sulphates","alcohol",'quality')
wineb <- wine %>%
  mutate(qualitybinary= ifelse(wine$quality<7, 0, 1)
  ) %>%
  select(-quality)
#a binárisan is nézhetjük, de ehhez a quality oszlopot törölni kell

wine$quality<-as.factor(wine$quality)
wineb$qualitybinary<-as.factor(wineb$qualitybinary)

index<-createDataPartition(wine$quality, p= .8, list=FALSE)

summary(wineb$qualitybinary)

#(az oszlopok átnevezése a véletlenerdőhöz kellett)

#Nézzük meg a megbecsülendő minőség értéket: 1382: 0, 217: 1. 


#Bagging:
  
set.seed(42) 
erdo <- randomForest(qualitybinary~., data=wineb,  importance=T)
keresztval <- trainControl(method="cv", number=10)
caret::train(qualitybinary~., data=wineb, method="rf", trControl=keresztval)

#91.99%-os pontosság (ami a (valós jól besoroltak az egyes kategóriákba)/ összes eset )és 0.59 a Kappa értéke.

#ábrázolás:
  
plot(erdo)
varImpPlot(erdo)


#Boosting:
 # Generalized Boosted Regression Modeling

set.seed(42) 
boost <- gbm(qualitybinary~., data=wineb,distribution="bernoulli")
caret::train(qualitybinary~., data=wineb, method="gbm", trControl=keresztval)

#Pontosság alapján: 88.4%-os pontosság, kappa: 0.4 (50fa, mélység 3, shrinkage: 0.1)



boost<-caret::train(qualitybinary~., data=wineb, method="gbm", trControl=keresztval)
fontossag_boost <- varImp(boost)
fontossag_boost$importance


#Stacking:
  
 # a) elsőként adat újboli előkészítés: 1,0-kat"Yes"/"No"-ra kell változtatni

#stackinghez
wines<-wineb
wines$qualitybinary<-as.numeric(wines$qualitybinary) #0,1-ből 1,2-t csinál
wines$qualitybinary[wines$qualitybinary == 1] <- "No"     
wines$qualitybinary[wines$qualitybinary == 2] <- "Yes" #minőségi



set.seed(42)
keresztval2 <- trainControl(method="cv", number=10,classProbs=TRUE)
modellek <- caretList(qualitybinary~., data=wines, trControl=keresztval2, methodList=c("knn", "rf","earth"))
stacking<-caretStack(modellek, method="glm", trControl=keresztval)
stacking


#90.9%-os pontosság, és a Kappa értékünk 0.562.
#(Bagging>Stacking>Boosting)

#Értelmezés döntési fával:
  
wineb$becsult<- predict(erdo, wineb)
ertelmezo_fa <- rpart(becsult~.-qualitybinary, data=wineb)
rpart.plot(ertelmezo_fa, type=3)

#Elsőként az alkohol százalék alapján bontjuk.


wineb <- wineb %>%
  select(-becsult)

wineb$becsult<- predict(boost, wineb)
ertelmezo_fa_boost <- rpart(becsult~.-qualitybinary, data=wineb)
rpart.plot(ertelmezo_fa_boost,type=2)



wineb <- wineb %>%
  select(-becsult)

wines$becsult<- predict(stacking, wines)
ertelmezo_fa_stacking <- rpart(becsult~.-qualitybinary, data=wines)
rpart.plot(ertelmezo_fa_stacking,type=2)



wines <- wines %>%
  select(-becsult)


#XGBOOST:
  
winex<-wines
winex$qualitybinary[winex$qualitybinary == "No"] <- 0    
winex$qualitybinary[winex$qualitybinary == "Yes"] <- 1
winex$qualitybinary<-as.numeric(winex$qualitybinary)
labelx<-as.numeric(unlist(winex[,12]))
winex<-as.matrix(winex[,-12])

#Yes/No-ra kellett állítani a bináris változót.


set.seed(42) 
xgb <- xgboost(data = winex, 
               label = labelx,
               eta = 0.5, #tanulási ráta
               nround=25, 
               subsample = 0.5, #random az adathalmaz felét haszáljuk, így nincs túlilleszkedés
               max.depth =6, #maximális mélység
               objective = "binary:logistic",
               nthread = 3)

#A változók fontossága hisztogramon ábrázolvaa legutóbbi XGBoost modellben

imp <-  xgb.importance(model = xgb)
xgb.plot.importance(imp, 
                    rel_to_first = T, 
                    xlab = 'Relative importance',
                    top_n = 11)




set.seed(42)
Train<-wine[index,]
Test<-wine[-index,]

xgb <- xgboost(data = winex[index,], 
               label = labelx[index],
               eta = 0.5, #tanulási ráta
               nround=25, 
               subsample = 0.5, #random az adathalmaz felét haszáljuk, így nincs túlilleszkedés
               max.depth =6, #maximális mélység
               objective = "binary:logistic",
               nthread = 3)

pred<-predict(xgb, winex[-index,])
err <- mean(as.numeric(pred > 0.5) != labelx[-index])
print(paste("test-error=", err))


pred_cat = as.factor(ifelse(pred > 0.5, "1", "0"))
confm<-confusionMatrix(pred_cat, as.factor(labelx[-index]), positive = "1")
confm


table <- data.frame(confm$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "jo", "rossz")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(jo = "green", rossz = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))

#Az ábra: az XGBOOST confusion mátrixa

#Nézzük meg külön a test, train spliten a legjobb modelltípust, a véletlen erdőt:
  
Train<-wineb[index,]
Test<-wineb[-index,]
erdo<-randomForest(Train$qualitybinary~., data=Train)
pred<-predict(erdo, Test, type="class")
confm<-confusionMatrix(pred, Test$quality)
confm


table <- data.frame(confm$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "jo", "rossz")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(jo = "green", rossz = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))



########################################################

#3-8 kategóriák, hogy összehasonlíthassuk a neurális hálókkal becsült többváltozós klasszifikációval:
 # Nézzük meg a megbecsülendő minőség értéket:
  
plot(wine$quality)


#Egy fa:
  
set.seed(42)
Train<-wine[index,]
Test<-wine[-index,]

fa<-rpart(Train$quality~., data=Train)
pred<-predict(fa, Test, type="class")
confm<-confusionMatrix(pred, Test$quality)
confm


#60.25% pontosság, kappa: 0.3359

table <- data.frame(confm$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "jo", "rossz")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(jo = "green", rossz = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))



#Erdő:
  
set.seed(42)
erdot<-randomForest(Train$quality~., data=Train)
pred<-predict(erdot, Test, type="class")
confm<-confusionMatrix(pred, Test$quality)
confm

#Pontosság: 72.87%, kappa: 0.5555. 

table <- data.frame(confm$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "jo", "rossz")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(jo = "green", rossz = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))



#Bagging:
  
set.seed(42) 
erdo <- randomForest(quality~., data=wine,  importance=T)
caret::train(quality~., data=wine, method="rf", trControl=keresztval)

#71%-os pontosság (ami a (valós jól besoroltak az egyes kategóriákba)/ összes eset )és 0.53 a Kappa értéke.

#Stochastic Gradient Boosting:
  
  
set.seed(42) 
boost <- gbm(quality~.,distribution = 'multinomial', data=wine)
caret::train(quality~., data=wine, method="gbm", trControl=keresztval)

#150 fából álló (pontosság 64.1%, kappa: 0.423)

#Nézzünk meg egy XGBoost-ot a több kategóriájú kimeneteli változónk becslésére

set.seed(42) 
wine<- read_csv("winequality-red.csv",show_col_types = FALSE) #a faktorok numerikussá válása során transzformálódnak
wine$quality <- as.numeric(wine$quality)
winem<-as.matrix(wine)
xgb <- xgboost(data = winem[,-12], 
               label = winem[,12],
               eta = 0.5, #tanulási ráta
               max_depth = 15, #maximális mélység
               nround=25, 
               subsample = 0.5, #random az adathalmaz felét haszáljuk, így nincs túlilleszkedés
               colsample_bytree = 0.5,
               eval_metric = "merror",
               #merror esetén minimalizálni akarjuk a rossz/összes esetet
               #ezt ajánlja multiclass probléma esetén
               objective = "multi:softprob",
               num_class = 10,
               nthread = 3)

#A [25]ös "futás" eredményeként már csak 0.3%-ot becsül rosszul. (felmerül a túlilleszkedés kérdése)

#(A Stacking caretben nincs megfelelően implementálva "multiclass" problémára.)