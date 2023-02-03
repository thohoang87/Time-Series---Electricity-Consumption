#vider la mémoire
rm(list=ls())

library(ggplot2)
library(forecast)
library(tseries)
library(keras)
library(readxl)
library(writexl)

# 1. Visualisation de la série et premières analyses

setwd("/Users/dangnguyenviet/Desktop/Master 2 SISE/cours/Séries temporelles et données séquentielles/projet")
df <- read_excel("Elec-train.xlsx")
str(df)
print(df)

# On scinde la série en train et prédiction:
df_train <- df[1:4507,2:3] #jusqu'au 16eme jour
df_pred <- df[4508:4603,3] # 17eme jour

# on transforme en série:
elect_train <- ts(df_train,start=c(1,5),end=c(16,96),frequency = 96)
elect_pred <- ts(df_pred,start=c(17,1),end=c(17,96),frequency = 96)

# présentation graphique:
autoplot(elect_train)+
  xlab('jour')+
  ylab('consommation')

ggseasonplot(elect_train[,"Power (kW)"],year.labels= TRUE,year.labels.left=TRUE)

# On scinde la série en apprentissage et test:
df_app <- window(elect_train,start=c(1,5),end=c(15,96))
df_test <- window(elect_train,start=c(16,1),end=c(16,96))

# présentation graphique:
plot(df_app)
plot(df_test)

################### Prédiction sans co-variables #############################

# 2. Premières prédictions avec Holt Winters
fit1 <- hw(df_app[,"Power (kW)"],damped=FALSE)
fit2 <- hw(df_app[,"Power (kW)"],damped=TRUE)
pred1 <- forecast(fit1,h=96)
pred2 <- forecast(fit2,h=96)
autoplot(df_test[,"`Power (kW)`"])+autolayer(pred1$mean,series="HW without covariate")
+autolayer(pred2$mean,series="HW+damped without covariate")
#ça ne fonctionne pas car la fréquence est trop élevée.

# 3. Prédictions avec SARIMA :
# 3.1. SARIMA par défaut
fit3 <- auto.arima(df_app[,"Power (kW)"])
summary(fit3)
pred3 <- forecast(fit3,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred3$mean,series="SARIMA by defaut without covariate")
print(sqrt(mean((pred3$mean-df_test[,"Power (kW)"])^2)))
#ARIMA(5,0,0)(0,1,0)[96] 
#9.018445

checkresiduals(fit3)
plot(pacf(fit3$residuals))
# 3.2.prédictions avec ARIMA(8,1,0)(0,1,1)
fit4 <- Arima(df_app[,"Power (kW)"],order=c(8,1,0),seasonal=list(order=c(0,1,1),period=96))
pred4 <- forecast(fit4,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred4$mean,series="SARIMA without covariate")

fit4 %>% residuals() %>% ggtsdisplay()
print(sqrt(mean((pred4$mean-df_test[,"Power (kW)"])^2)))
# 8.381082

# 4. NN par défaut:
fit5 <- nnetar(df_app[,"Power (kW)"])
print(fit5)
#NNAR(19,1,10)[96] 
pred5 <- forecast(fit5,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred5$mean,series="NN by defaut without covariate")
print(sqrt(mean((pred5$mean-df_test[,"Power (kW)"])^2)))
#51.24026
fit5 %>% residuals() %>% ggtsdisplay()

# on modifie les paramètres 
fit6 <- nnetar(df_app[,"Power (kW)"],p=19,P=96,k=8)
pred6 <- forecast(fit6,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred6$mean,series="NN without covariate")
print(sqrt(mean((pred6$mean-df_test[,"Power (kW)"])^2)))
#17.94947
fit6 %>% residuals() %>% ggtsdisplay()

#zoom sur la prédiction
plot(df_test[,"Power (kW)"])
lines(pred4$mean,col=2)
lines(pred6$mean,col=3)
legend('topleft',col=1:3,lty=1,legend=c('initial','previsions SARIMA','previsions NN'))

################### Prédiction avec co-variables #############################

# 5. SARIMA par défaut avec co-variable
fit7 <- auto.arima(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C°)"])
summary(fit7)
#ARIMA(5,1,1)(0,1,0)[96]
pred7 <- forecast(fit7,h=96,xreg=df_test[,"Temp (C°)"])
autoplot(df_test)+autolayer(pred7$mean)
print(sqrt(mean((pred7$mean-df_test[,"Power (kW)"])^2)))
#8.492825

fit7 %>% residuals() %>% ggtsdisplay()

plot(df_app[,"Power (kW)"],df_app[,"Temp (C°)"])

# on modifie les paramètres:
fit8 <- Arima(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C°)"],order=c(8,1,1),seasonal = c(0,1,1))
checkresiduals(fit8)
pred8 <- forecast(fit8,h=96,xreg=df_test[,"Temp (C°)"])
autoplot(df_test)+autolayer(pred8$mean)
print(sqrt(mean((pred8$mean-df_test[,"Power (kW)"])^2)))
#7.443478

# 6. NN par défaut avec co-variable

fit9 <- nnetar(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C°)"])
print(fit9)
#NNAR(19,1,11)[96] 
pred9 <- forecast(fit9,xreg=df_app[,"Temp (C°)"],h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred9$mean,series="NN by defaut without covariate")
print(sqrt(mean((pred9$mean-df_test[,"Power (kW)"])^2)))
#18.75048
fit9 %>% residuals() %>% ggtsdisplay()

# on modifie les paramètres 
fit10 <- nnetar(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C°)"],p=19,P=96,k=11)
pred10 <- forecast(fit10,xreg=df_app[,"Temp (C°)"],h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred10$mean,series="NN without covariate")
print(sqrt(mean((pred10$mean-df_test[,"Power (kW)"])^2)))
#18.15401
fit10 %>% residuals() %>% ggtsdisplay()

#zoom sur la prédiction avec co-variable
plot(df_test[,"Power (kW)"])
lines(pred8$mean,col=2)
lines(pred10$mean,col=3)
legend('topleft',col=1:3,lty=1,legend=c('initial','previsions SARIMA avec co-variable','previsions NN avec co-variable'))

################### Construction les modèles avec et sans co-variables #############################
# sans co-variables avec ARIMA(8,1,0)(0,1,1)
fit11 <- Arima(elect_train[,"Power (kW)"],order=c(8,1,0),seasonal=list(order=c(0,1,1),period=96))

#avec co-variables avec ARIMA(8,1,1)(0,1,1)
fit12 <- Arima(elect_train[,"Power (kW)"],xreg=elect_train[,"Temp (C°)"],order=c(8,1,1),seasonal = c(0,1,1))

################### Prédiction pour 2/17/2010  avec et sans co-variables #############################
# sans co-variables:
pred11 <- forecast(fit11,h=96)
autoplot(elect_train[,"Power (kW)"])+autolayer(pred11$mean)
lst1<-data.frame(pred11$mean)
str(lst1)
write_xlsx(lst1, "prédiction_sans_covariables.xlsx")

#avec co-variables:
pred12 <- forecast(fit12,h=96,xreg=elect_pred)
autoplot(elect_train)+autolayer(pred12$mean)
lst2<-data.frame(pred12$mean)
str(lst2)
write_xlsx(lst2, "prédiction_avec_covariables.xlsx")
