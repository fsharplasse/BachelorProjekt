## Bachelorprojekt

library(ggplot2)
library(tidyverse)
library(caTools)
library(dplyr)
library(ROCR)
library(rmarkdown)
library("readxl")

#Nedenstående kode indlæser de to datafiler, og giver dem en funktion.
bilbasen <- read_delim("bilbasen_scrape.csv", 
                       delim = ";", escape_double = FALSE, locale = locale(encoding = "WINDOWS-1252"), 
                       trim_ws = TRUE)

choice <- read_delim("choice_data.csv", 
                     delim = ";", escape_double = FALSE, locale = locale(encoding = "WINDOWS-1252"), 
                     trim_ws = TRUE)

#Indlæser den kombinerede fil med de valgte variabler
# final <- read_excel("final_data.xlsx")

final <- read_delim("finale.csv", 
                     delim = ",", escape_double = FALSE, locale = locale(encoding = "WINDOWS-1252"), 
                     trim_ws = TRUE)
#Nedenstående kode ændre variabelnavnene i datasættet "final":
colnames(final)[1] <- "No"
colnames(final)[7] <- "Weight"
colnames(final)[8] <- "EngineEffect"
colnames(final)[6] <- "Segment"
colnames(final)[10] <- "CostKM"

#Nedenstående kode ændre variabelnavnene i datasættet "choice":
colnames(choice)[12] <- "Weight"
colnames(choice)[16] <- "Size"
colnames(choice)[14] <- "EngineEffect"
colnames(choice)[10] <- "Prices"
colnames(choice)[7] <- "FuelSize"
colnames(choice)[1] <- "MakeYear"
colnames(choice)[8] <- "NoRegistrations"
colnames(choice)[2] <- "MakeModelYear"

choice_sub <- choice %>% group_by(MakeYear, Fuel) %>% summarise(AverageWeight = mean(Weight))

ggplot(choice_sub,aes(x=MakeYear,y=AverageWeight,color=Fuel))+
  xlab("Registration Year") +
  ylab("Weight in Kg")+
  geom_line()

choice_sub1 <- choice %>% group_by(MakeYear, Fuel) %>% summarise(AverageSize = mean(Size))

ggplot(choice_sub1,aes(x=MakeYear,y=AverageSize,color=Fuel))+
  xlab("Registration Year") +
  ylab("Size in cubic metres")+
  geom_line()

choice_sub2 <- choice %>% group_by(MakeYear, Fuel) %>% summarise(AverageEE = mean(EngineEffect))

ggplot(choice_sub2,aes(x=MakeYear,y=AverageEE,color=Fuel))+
  xlab("Registration Year") +
  ylab("Engine Effect (kW)")+
  geom_line()

choice_sub3 <- choice %>% group_by(MakeYear, FuelSize) %>% summarise(AveragePrice = mean(Prices))

ggplot(choice_sub3,aes(x=MakeYear,y=AveragePrice,color=FuelSize))+
  xlab("Registration Year") +
  ylab("Price")+
  geom_line()

eldummy <- ifelse(choice$Fuel == "El",1,0)
didummy <- ifelse(choice$Fuel == "Diesel",1,0)
model <-lm(formula = Shares ~ Weight + EngineEffect + Prices + eldummy + didummy, 
           data = choice)
summary(model)


install.packages("mlogit")    
library(mlogit)
## choice$FuelSize <- as.factor(choice$FuelSize)
## nested.logit <- mlogit(Shares~ Weight + EngineEffect + Prices + eldummy + didummy, choice,
##                       shape = "wide", nests = list(FuelSize = c('ElLarge', 'ElSmall'), 
##                                                 other = c('BenzinLarge', 'BenzinSmall', 'DieselLarge', 'DieselSmall')), un.nest.el = TRUE)
## summary(nested.logit)
## summary(choice$Shares)
## max(choice$Shares)
cho4 <- choice %>% group_by(MakeYear, Fuel) %>% summarise(TShares = sum(Shares))
ggplot(cho4,aes(x=MakeYear,y=TShares,color=Fuel))+
  xlab("Registration Year") +
  ylab("Share")+
  geom_line()

choice_sub5 <- choice %>% group_by(MakeYear, Fuel) %>% summarise(NoR = sum(NoRegistrations)) %>% mutate(Percentage = NoR/sum(NoR)*100)
## choice_sub5 %>% mutate(Percentage = NoR/sum(NoR))

ggplot(choice_sub5,aes(x=MakeYear,y=Percentage,color=Fuel))+
  xlab("Registration Year") +
  ylab("Share")+
  geom_line()

logit_reg<-glm(El ~ Weight + kmL + Shares + EngineEffect + Nypris + Diesel, 
               data = final, family = binomial(link = "logit"))
summary(logit_reg)

## if choice$model = bilbasen$model then kmL from choice also kmL in bilbasen

choice_sub6 <- data.frame("MakeModelYear" = choice$MakeModelYear) 

bilbasen$kmL<-gsub(",",".",as.character(bilbasen$kmL))
bilbasen$kmL<-gsub("-","NA",as.character(bilbasen$kmL))
bilbasen$kmL<-gsub("km","",as.character(bilbasen$kmL))
bilbasen$kmL<-gsub("/l","",as.character(bilbasen$kmL))
bilbasen$kmL<-gsub("(NEDC)","",as.character(bilbasen$kmL))
bilbasen$kmL<-gsub("[()]","",as.character(bilbasen$kmL))
bilbasen$kmL<-as.numeric(bilbasen$kmL)
bilbasen[!is.na(bilbasen$kmL), ]
bilbasen$kmL <-as.factor(bilbasen$kmL)
bilbasen_sub1 <- bilbasen %>% group_by(make_model, aargang, drivkraft) %>% summarise(AveragekmL = mean(kmL))


final[!is.na(final$kmL), ]
cbind(
  lapply(
    lapply(final, is.na)
    , sum)
)

final$kmL<-gsub("km","",as.character(final$kmL))
final$kmL<-gsub("/l","",as.character(final$kmL))
final$kmL<-gsub("(NEDC)","",as.character(final$kmL))
final$kmL<-gsub("[()]","",as.character(final$kmL))
final$kmL<-gsub(".",",",as.character(final$kmL))
final$kmL<-gsub("-","145.5",as.character(final$kmL))

final$kmL<-as.numeric(final$kmL)
final$nypris_kr<-as.numeric(final$nypris_kr)

summary(glm(Fuel ~ Nypris + Weight + Shares + EngineEffect + kmL, data = final, family=binomial))

str(final)

#ElD <- ifelse(final$Fuel == "El",1,0)
#DieselD <- ifelse(final$Fuel == "Diesel",1,0)

model <-lm(formula = Shares ~ Weight + EngineEffect + Nypris + ElD + DieselD + kmL, 
           data = final)
summary(model)
print(summary(model),digits=10) 

modelo <-lm(formula = Shares ~ Nypris, 
           data = final)
summary(modelo)


YearOld <- 2022 - final$Year

#### Multnominal logit #####
final[complete.cases(final), ]
final$Fuel = as.factor(final$Fuel)
final$Segment = as.factor(final$Segment)
final$Make = as.factor(final$Make)
## mlogit pakken
library("mlogit")
mldata = mlogit.data(final,choice = 'Fuel', shape = "wide")
summary(mldata)
mlogit.model1 <- mlogit(Segment ~ 0 | CostKM, data = mldata,
                        nests = list(size = c('ElSmall', 'ElBig'), other = c('BenzinSmall', 'BenzinBig'), other2 = c('DieselSmall', 'DieselBig')), un.nest.el = TRUE)
summary(mlogit.model1)

summary(mlogit(formula = Fuel ~ 0 | Nypris + Weight + EngineEffect, data = mldata, reflevel="Benzin", print.level = 0))

exp(coefficients(mlogit.model1))

## nnet pakken
library("nnet")
summary(multinom(Make ~ YearOld + Nypris + CostKM + Weight + EngineEffect, data = final, maxit = 300))

### LOGIT ###

## glm
logit_reg<-glm(El ~ YearOld + Nypris + Weight + EngineEffect, 
               data = final, family = binomial(link = "logit"))
summary(logit_reg)

logit_reg<-glm(El ~ YearOld + Weight + Shares + EngineEffect + Nypris + Diesel, 
               data = final, family = binomial(link = "logit"))
summary(logit_reg)

table(final$Make)
