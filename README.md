# Time-Series---Electricity-Consumption

## 1 Introduction
Le fichier Elec-train.xlsx contient la consommation d’électricité (kW) et la température extérieur d’un bâtiment. Ces grandeurs sont mesurées toutes les 15 minutes, du 01/01/2010 1h15 au 16/02/2010 23h45. De plus, la température extérieur est disponible pour le 17/02/2010.

L’objectif est de prévoir la consommation d’électricité (kW) pour le 17/02/2010, puis d’obtenir la meilleure prévision possible. Il faut donc tester tous les modèles vus pendant le cours, les régler et les comparer correctement.

Par conséquence, les modèles que j’ai utilisé sont :

— Modèle Holt Winters

— Modèle SARIMA

— Modèle Neural network


## 2 Visualisation de la série et premières analyses

Dans un premier temps, on scinde la série originale en ”train” et ”prédiction”. La partie ”train” correspond à la consommation d’électricité (kW) et la température extérieur du 01/01/2010 1h15 au 16/02/2010 23h45. Et la partie ”prédiction” correspond à la température extérieur pour le 17/02/2010. 

Le choix de la fréquence :
On voit que 1 jour = 1440 minutes = 15 minutes * 96. Donc j’ai choisit fréquence = 96.

```
library(ggplot2)
library(forecast)
library(tseries)
library(keras)
library(readxl)
library(writexl)
```

On sépare la série en train et prédiction: 
```
df_train <- df[1:4507,2:3]
df_pred <- df[4508:4603,3]
```

On transforme en série:
```
elect_train <- ts(df_train,start=c(1,5),end=c(16,96),frequency = 96) 
elect_pred <- ts(df_pred,start=c(17,1),end=c(17,96),frequency = 96)
```
Puis on fait une présentation graphique :
```
autoplot(elect_train)+
  xlab(’jour’)+
  ylab(’consommation’)
```
<img width="1440" alt="donnees" src="https://user-images.githubusercontent.com/114235978/216565743-fdda1374-b0e1-4a75-8095-1d5789fcee3f.png">

```
 ggseasonplot(elect_train[,"Power (kW)"],year.labels= TRUE,year.labels.left=TRUE)
 ```
 <img width="916" alt="saison" src="https://user-images.githubusercontent.com/114235978/216566657-d8879306-27b5-4974-b06e-57f870f6ae10.png">

On voit que la série n’a pas de tendance, mais elle a une saisonnalité. On observe que la consommation d’électricité est faible le matin, puis elle augmente vers l’après-midi, elle atteint son pique le soir, et elle descend vers la nuit.

Enfin, on scinde la série en apprentissage et test :
```
df_app <- window(elect_train,start=c(1,5),end=c(15,96))
df_test <- window(elect_train,start=c(16,1),end=c(16,96))
```

## 3 Prédiction sans température
### 3.1 Premières prédictions avec Holt Winters
```
fit1 <- hw(df_app[,"Power (kW)"],damped=FALSE)
fit2 <- hw(df_app[,"Power (kW)"],damped=TRUE)
```
Cepedant, ce modèle n’est pas adapté car la fréquence est trop élevée.

### 3.2 Prédictions avec SARIMA

Modèle SARIMA par défaut
```
fit3 <- auto.arima(df_app[,"Power (kW)"])
summary(fit3)
```
<img width="394" alt="sumfit3" src="https://user-images.githubusercontent.com/114235978/216566739-096b6f31-f818-42e5-87f8-09494874aec9.png">

```
pred3 <- forecast(fit3,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred3$mean,series="SARIMA by defaut without covariate")
```
<img width="1440" alt="predfit3" src="https://user-images.githubusercontent.com/114235978/216566782-7c3e8b26-f803-4b94-b6ce-6925da159aed.png">

```
print(sqrt(mean((pred3$mean-df_test[,"Power (kW)"])^2)))
## [1] 9.018445
```
On vérifie les résidus :
```
checkresiduals(fit3)
plot(pacf(fit3$residuals))
```
<img width="1440" alt="resfit3" src="https://user-images.githubusercontent.com/114235978/216566813-24c7e4b6-cae1-437f-9cba-9a103ce3823a.png">

On voit que sur le graphique ACF, il y a un grand pique à lag 96, c-a-d il y a bien la saisonnalité. Et sur le graphique PACF, le plus grand pique est à lag 8 et à lag 13. Au final, je choisit à tester le modèle ARIMA(8,1,0)(0,1,1).
 
Modèle ARIMA(8,1,0)(0,1,1)
```
fit4 <- Arima(df_app[,"Power (kW)"],order=c(8,1,0), seasonal=list(order=c(0,1,1),period=96))
pred4 <- forecast(fit4,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred4$mean,series="SARIMA without covariate")
```
<img width="1440" alt="sarimafit4" src="https://user-images.githubusercontent.com/114235978/216566971-e79ad824-0e97-4aff-b46b-8de65c5a25b0.png">

```
fit4 %>% residuals() %>% ggtsdisplay()
```
<img width="1440" alt="resfit4" src="https://user-images.githubusercontent.com/114235978/216566992-3812c759-74a5-4fd2-8359-c3c20cc1d344.png">

```
print(sqrt(mean((pred4$mean-df_test[,"Power (kW)"])^2)))
## [1] 8.381082
```
On voit que ce modèle donne le meilleur résultat que le modèle SARIMA par défaut.

### 3.3 Prédictions avec Neural Network

Modèle NN par défaut
```
fit5 <- nnetar(df_app[,"Power (kW)"])
pred5 <- forecast(fit5,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred5$mean,series="NN by defaut without covariate")
```
<img width="1440" alt="nndefaut" src="https://user-images.githubusercontent.com/114235978/216567094-d67571ca-266b-4c67-980e-b791402dd8af.png">


```
print(sqrt(mean((pred5$mean-df_test[,"Power (kW)"])^2)))
## [1] 41.59443
```
```
fit5 %>% residuals() %>% ggtsdisplay()
```
<img width="1440" alt="resfit5" src="https://user-images.githubusercontent.com/114235978/216567128-62b51214-8c75-47bb-bd87-45ddc413c4a2.png">

Modèle NN avec les paramètres modifiés
```
fit6 <- nnetar(df_app[,"Power (kW)"],p=19,P=96,k=8)
pred6 <- forecast(fit6,h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred6$mean,series="NN without covariate")
```
<img width="1440" alt="nnfit6" src="https://user-images.githubusercontent.com/114235978/216567181-aeef6bcd-96ef-4e9c-bb81-0784482d8492.png">

```
print(sqrt(mean((pred6$mean-df_test[,"Power (kW)"])^2)))
## [1] 27.45869
```
```
fit6 %>% residuals() %>% ggtsdisplay()
```
<img width="1440" alt="resfit6" src="https://user-images.githubusercontent.com/114235978/216567211-b0a44b61-607a-461c-8915-d3fcda7a4bf1.png">

Comparaison entre le modèle SARIMA et NN
On voit bien que le modèle SARIMA donne le meilleur résultat que le modèle NN. On choisit donc ce modèle.

## 4 Prédiction avec température

On regarde s’il y a la corrélation entre la consommation d’électricité et la température :
```  
plot(df_app[,"Power (kW)"],df_app[,"Temp (C)"])
```
<img width="1199" alt="correlation" src="https://user-images.githubusercontent.com/114235978/216567257-16c931f4-2675-408d-b524-975e5f46df23.png">

On voit qu’il n’y a pas vraiment de corrélation entre ces 2 variables.
 
### 4.1 Prédictions avec SARIMA

Modèle SARIMA par défaut
```
fit7 <- auto.arima(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C)"])
summary(fit7)
```
<img width="481" alt="sumfit7" src="https://user-images.githubusercontent.com/114235978/216567299-5b8b3eba-fc33-480c-bc2b-483228629038.png">

```
pred7 <- forecast(fit7,h=96,xreg=df_test[,"Temp (C)"])
autoplot(df_test)+autolayer(pred7$mean)
```
<img width="1440" alt="predfit7" src="https://user-images.githubusercontent.com/114235978/216567333-7b72b995-49f3-4dad-b30b-22f7aeb170ea.png">

```
print(sqrt(mean((pred7$mean-df_test[,"Power (kW)"])^2)))
## [1] 8.492825
```
On vérifie les résidus :
```
fit7 %>% residuals() %>% ggtsdisplay()
```
<img width="1440" alt="resfit7" src="https://user-images.githubusercontent.com/114235978/216567369-7585fe72-27ae-4591-97b1-78cd6987d9d7.png">

Modèle ARIMA(8,1,1)(0,1,1)
```
fit8 <- Arima(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C)"], order=c(8,1,1),seasonal = c(0,1,1))
pred8 <- forecast(fit8,h=96,xreg=df_test[,"Temp (C)"])
autoplot(df_test)+autolayer(pred8$mean)
```
<img width="1440" alt="sarimafit8" src="https://user-images.githubusercontent.com/114235978/216567424-2e409687-beab-4dce-82fc-9c2ec18a8b2d.png">

```
checkresiduals(fit8)
```
<img width="1440" alt="resfit8" src="https://user-images.githubusercontent.com/114235978/216567443-e0271754-5595-46b2-ad5d-5099ea494967.png">

```
print(sqrt(mean((pred8$mean-df_test[,"Power (kW)"])^2)))
## [1] 7.443478
```
On voit bien que le modèle ARIMA(8,1,1)(0,1,1) donne le meilleur résultat que le modèle SARIMA par défaut.

### 4.2 Prédictions avec Neuron Network

Modèle NN par défaut
```
fit9 <- nnetar(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C)"])
pred9 <- forecast(fit9,xreg=df_app[,"Temp (C)"],h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred9$mean,series="NN by defaut without covariate")
```
<img width="1440" alt="nndefaut9" src="https://user-images.githubusercontent.com/114235978/216567469-39ff631b-81ed-4746-b278-e34f815ca20c.png">

```
print(sqrt(mean((pred9$mean-df_test[,"Power (kW)"])^2)))
## [1] 18.75048
```
```
fit9 %>% residuals() %>% ggtsdisplay()
```
<img width="1440" alt="resfit9" src="https://user-images.githubusercontent.com/114235978/216567512-a262531f-c0ab-453f-8eb3-5f17b21143d6.png">

Modèle NN avec les paramètres modifiés
```
fit10 <- nnetar(df_app[,"Power (kW)"],xreg=df_app[,"Temp (C)"],p=19,P=96,k=11)
pred10 <- forecast(fit10,xreg=df_app[,"Temp (C)"],h=96)
autoplot(df_test[,"Power (kW)"])+autolayer(pred10$mean,series="NN without covariate")
```
<img width="1440" alt="nnfit10" src="https://user-images.githubusercontent.com/114235978/216567550-70c8daaf-d4d7-44b5-bd5d-c6e0a9d0f70f.png">

```
print(sqrt(mean((pred10$mean-df_test[,"Power (kW)"])^2)))
## [1] 18.15401
```
```
fit10 %>% residuals() %>% ggtsdisplay()
```
<img width="1440" alt="resfit10" src="https://user-images.githubusercontent.com/114235978/216567572-32102b4b-18f3-4122-956a-e0ed5ea99de9.png">

Comparaison entre le modèle SARIMA et NN

On voit que le modèle SARIMA donne le meilleur résultat que le modèle NN. On choisit donc ce modèle pour faire la prédiction pour le 17/02/2010.

## 5 Prédiction pour le 17/02/2010 
### 5.1 Prédictions sans température
```
fit11 <- Arima(elect_train[,"Power (kW)"], order=c(8,1,0),seasonal=list(order=c(0,1,1),period=96)) 
pred11 <- forecast(fit11,h=96)
autoplot(elect_train[,"Power (kW)"])+autolayer(pred11$mean) 
lst1<-data.frame(pred11$mean)
write_xlsx(lst1, "pr ́ediction_sans_covariables.xlsx")
```
<img width="1146" alt="prediction_sans_covariables" src="https://user-images.githubusercontent.com/114235978/216567633-9709521f-221f-4397-bd6f-cb29d46937c8.png">

### 5.2 Prédictions avec température
 ```
fit12 <- Arima(elect_train[,"Power (kW)"],xreg=elect_train[,"Temp (C)"],order=c(8,1,1),seasonal = c(0,1,1))
pred12 <- forecast(fit12,h=96,xreg=elect_pred)
autoplot(elect_train)+autolayer(pred12$mean)
lst2<-data.frame(pred12$mean)
write_xlsx(lst2, "pr ́ediction_avec_covariables.xlsx")
```
<img width="1148" alt="prediction_avec_covariables" src="https://user-images.githubusercontent.com/114235978/216567667-735d17b2-f9a0-4d17-a3ac-7be0301acf79.png">

## 6 Conclusion

Pour le modèle sans température, le meilleur modèle est ARIMA(8,1,0)(0,1,1)[96], qui donne un taux d’erreur 8.381082.

Pour le modèle avec température, le meilleur modèle est ARIMA(8,1,1)(0,1,1)[96], qui donne un taux d’erreur 7.443478.

Si on compare ces deux modèles, on voit que le modèle avec température donne le meilleur résultat que le modèle sans température.
