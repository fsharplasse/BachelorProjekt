## Bachelorprojekt

library(ggplot2)
library(tidyverse)
library(caTools)
library(dplyr)
library(ROCR)
library(rmarkdown)

bilbasen <- read_delim("bilbasen_scrape.csv", 
                       delim = ";", escape_double = FALSE, locale = locale(encoding = "WINDOWS-1252"), 
                       trim_ws = TRUE)

choice <- read_delim("choice_data.csv", 
                     delim = ";", escape_double = FALSE, locale = locale(encoding = "WINDOWS-1252"), 
                     trim_ws = TRUE)

colnames(choice)[12] <- "Weight"
colnames(choice)[16] <- "Size"
colnames(choice)[14] <- "EngineEffect"
colnames(choice)[10] <- "Prices"
colnames(choice)[7] <- "FuelSize"
colnames(choice)[1] <- "MakeYear"
colnames(choice)[8] <- "NoRegistrations"

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

## summary(choice$Shares)
## max(choice$Shares)
cho4 <- choice %>% group_by(MakeYear, Fuel) %>% summarise(TShares = sum(Shares))
ggplot(cho4,aes(x=MakeYear,y=TShares,color=Fuel))+
  xlab("Registration Year") +
  ylab("Share")+
  geom_line()

choice_sub5 <- choice %>% group_by(MakeYear, Fuel) %>% summarise(NoR = sum(NoRegistrations)) %>% mutate(Percentage = NoR/sum(NoR))
## choice_sub5 %>% mutate(Percentage = NoR/sum(NoR))

ggplot(choice_sub5,aes(x=MakeYear,y=Percentage,color=Fuel))+
  xlab("Registration Year") +
  ylab("Share")+
  geom_line()

