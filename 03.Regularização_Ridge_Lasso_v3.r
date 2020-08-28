
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#   _____                  _            _               /\/|       
#  |  __ \                | |          (_)             |/\/        
#  | |__) |___  __ _ _   _| | __ _ _ __ _ ______ _  ___ __ _  ___  
#  |  _  // _ \/ _` | | | | |/ _` | '__| |_  / _` |/ __/ _` |/ _ \ 
#  | | \ \  __/ (_| | |_| | | (_| | |  | |/ / (_| | (_| (_| | (_) |
#  |_|  \_\___|\__, |\__,_|_|\__,_|_|  |_/___\__,_|\___\__,_|\___/ 
#               __/ |                               )_)            
#              |___/                                               
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

rm(list=ls())
library(glmnet)
library(dplyr)
library(tidyverse)
library(caret)
library(gridExtra)
library(plotROC)
library(grid)
library(gridExtra)
library(MASS)

set.seed(999)

# ----------------------------------------------------------------------------
# Importando 

base0 <- read.csv("blogData_train.csv", sep = ",",
                  header = F) 

base0$V281    <- ifelse(base0$V281 > 0, 1,0)
base          <- base0[sample(nrow(base0), 3000), ]

base_val      <- base0[sample(nrow(base0), 1000), ]
base_val$V281 <- ifelse(base_val$V281 > 0, 1,0)

rm(base0)

# ----------------------------------------------------------------------------
# Separar treino e teste

training.samples <- base$V281 %>% createDataPartition(p = 0.8, list = FALSE)

train.data  <- base[training.samples, ]
train.data$V281 %>% table()

train.data1  <- train.data
train.data1$V281 %>% table()
train.data1$V281 %>% table() %>% prop.table()

test.data   <- base[-training.samples, ]

drop.cols <- c("V281")

x <- train.data1[,!(names(train.data1) %in% drop.cols)]
X <- as.matrix(x)
y <- train.data1$V281

x_test <- test.data[,!(names(test.data) %in% drop.cols)]
X_test <- as.matrix(x_test)
y_test <- test.data$V281

x_val <- base_val[,!(names(base_val) %in% drop.cols)]
X_val <- as.matrix(x_val)
y_val <- base_val$V281

y_test %>% table()
y_val  %>% table()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tuning para lambda para Lasso

# alpha=1 for the lasso penalty
cv.lasso <- cv.glmnet(X, y, type.measure= "auc", alpha = 1, family = "binomial")

db.lasso <- as.data.frame(cbind(cv.lasso$lambda, cv.lasso$cvm, cv.lasso$cvlo, cv.lasso$cvup))

gglasso <- ggplot(db.lasso, aes(x = log(V1), y = V2)) + 
  geom_errorbar(aes(ymin = V3, ymax = V4),colour = "#808080") +
  geom_point(colour = "#990033") + ggtitle("Lasso") + ylab("") + 
  xlab(expression(paste("log(", lambda, ")"))) +
  geom_vline(xintercept = log(cv.lasso$lambda.min), color = "#808080", linetype = "dashed") +
  geom_vline(xintercept = log(cv.lasso$lambda.1se), color = "#808080", linetype = "dashed")
#ggsave("lasso2.png", gglasso, 
#       width = 7, height = 5)  

cv.lasso$lambda.min
cv.lasso$lambda.1se
cv.lasso$lambda.min2 <- quantile(cv.lasso$lambda, 0.35)
coef(cv.lasso, cv.lasso$lambda.min)

# Modelo final Lasso 
lasso_model <- glmnet(X, y, alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.min2)

table(coef(lasso_model, s = 0) %>% as.vector() != 0)

vars_lasso <- table(lasso_model$beta %>% as.vector() != 0)
vars_lasso

pred_lasso <- predict(lasso_model, X_test, type ="response")
pred_lasso_val <- predict(lasso_model, X_val, type ="response")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tuning para lambda para Ridge

# alpha = 0 for the ridge penalty
cv.ridge <- cv.glmnet(X, y, type.measure= "auc", alpha = 0, family = "binomial")

db.ridge <- as.data.frame(cbind(cv.ridge$lambda, cv.ridge$cvm, cv.ridge$cvlo, cv.ridge$cvup))

ggridge <- ggplot(db.ridge, aes(x = log(V1), y = V2)) + 
  geom_errorbar(aes(ymin = V3, ymax = V4),colour = "#808080") +
  geom_point(colour = "#990033") + ggtitle("Ridge") +
  ylab("AUC") + xlab("") +
  geom_vline(xintercept = log(cv.ridge$lambda.min), color = "#808080", linetype = "dashed") +
  geom_vline(xintercept = log(cv.ridge$lambda.1se), color = "#808080", linetype = "dashed")
#ggsave("ridge2.png", ggridge, 
#       width = 7, height = 5)  

cv.ridge$lambda.min
cv.ridge$lambda.1se
coef(cv.ridge, cv.ridge$lambda.min)

# Modelo final Ridge 
ridge_model <- glmnet(X, y, alpha = 0, family = "binomial",
                      lambda = cv.ridge$lambda.min)

vars_ridge <- table(ridge_model$beta %>% as.vector() != 0)

pred_ridge <- predict(ridge_model, X_test, type ="response")
pred_ridge_val <- predict(ridge_model, X_val, type ="response")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tuning para Elastic Net

alpha <- 0.7 # seq(0.1, 1, 0.1)

# alpha = 0 for the ridge penalty
cv.elastic <- cv.glmnet(X, y, type.measure= "auc", alpha = alpha, family = "binomial")

db.elastic <- as.data.frame(cbind(cv.elastic$lambda, cv.elastic$cvm, cv.elastic$cvlo, cv.elastic$cvup))

ggelastic <- ggplot(db.elastic, aes(x = log(V1), y = V2)) + 
  geom_errorbar(aes(ymin = V3, ymax = V4),colour = "#808080") +
  geom_point(colour = "#990033") + ggtitle("ElasticNet") +
  ylab("") + xlab("") +
  geom_vline(xintercept = log(cv.elastic$lambda.min), color = "#808080", linetype = "dashed") +
  geom_vline(xintercept = log(cv.elastic$lambda.1se), color = "#808080", linetype = "dashed")
#ggsave("elastic2.png", ggelastic, 
#       width = 7, height = 5)  

cv.elastic$lambda.min2 <- min(cv.elastic$lambda)

coef(cv.elastic, cv.elastic$lambda.min)

# Modelo final Elastic Net 
elastic_model <- glmnet(X, y, alpha = alpha, family = "binomial",
                        lambda = cv.elastic$lambda.min)

vars_elastic <- table(elastic_model$beta %>% as.vector() != 0)
vars_elastic

pred_elastic <- predict(elastic_model, X_test, type ="response")
pred_elastic_val <- predict(elastic_model, X_val, type ="response")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# RegressÃ£o Logistica

logit <- glm(V281 ~ ., 
             data = train.data1,
             family = "binomial")
summary(logit)

step.model <- logit %>% stepAIC(trace = FALSE)
summary(step.model)

coefs <- summary(step.model)
vars_logit <- table(coefs$coefficients[,4] < 0.05)

pred_logit <- predict(step.model, x_test, type ="response")
pred_logit_val <- predict(step.model, x_val, type ="response")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ROC curves

df_roc_lasso <- cbind(y_test, pred_lasso, rep("Lasso",length(y_test)))     %>% as.data.frame()
names(df_roc_lasso) <- c("y","pred","mod")
df_roc_ridge <- cbind(y_test, pred_ridge, rep("Ridge",length(y_test)))     %>% as.data.frame()
names(df_roc_ridge) <- c("y","pred","mod")
df_roc_elast <- cbind(y_test, pred_elastic, rep("Elastic",length(y_test))) %>% as.data.frame()
names(df_roc_elast) <- c("y","pred","mod")
df_roc_logit <- cbind(y_test, pred_logit, rep("Logit",length(y_test)))     %>% as.data.frame()
names(df_roc_logit) <- c("y","pred","mod")

df_roc      <- rbind(df_roc_ridge,df_roc_lasso, df_roc_elast,df_roc_logit) %>% as.data.frame() 
df_roc$y    <- df_roc$y %>% as.character() %>% as.numeric()
df_roc$pred <- df_roc$pred %>% as.character() %>% as.numeric()
df_roc$base <- rep("Teste",length(dim(df_roc)[1]))

df_roc_lasso_val <- cbind(y_val, pred_lasso_val, rep("Lasso",length(y_val)))     %>% as.data.frame()
names(df_roc_lasso_val) <- c("y","pred","mod")
df_roc_ridge_val <- cbind(y_val, pred_ridge_val, rep("Ridge",length(y_val)))     %>% as.data.frame()
names(df_roc_ridge_val) <- c("y","pred","mod")
df_roc_elast_val <- cbind(y_val, pred_elastic_val, rep("Elastic",length(y_val))) %>% as.data.frame()
names(df_roc_elast_val) <- c("y","pred","mod")
df_roc_logit_val <- cbind(y_val, pred_logit_val, rep("Logit",length(y_val)))     %>% as.data.frame()
names(df_roc_logit_val) <- c("y","pred","mod")

df_roc_val      <- rbind(df_roc_ridge_val,df_roc_lasso_val, df_roc_elast_val,df_roc_logit_val) %>% as.data.frame() 
df_roc_val$y    <- df_roc_val$y %>% as.character() %>% as.numeric()
df_roc_val$pred <- df_roc_val$pred %>% as.character() %>% as.numeric()
df_roc_val$base <- rep("Validação",length(dim(df_roc_val)[1]))

df_roc0 <- rbind(df_roc,df_roc_val) %>% as.data.frame()

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

aux <- "Validação" # "Validação" "Teste"
df_roc00 <- df_roc0 %>% subset(base == aux)

roc <- ggplot(df_roc00, aes(d = y, m = pred, color  = mod)) + 
  geom_roc(increasing = T, labels = F, n.cuts = 0) +
  style_roc(theme = theme_grey, xlab = "(1 - Especificidade)", ylab = "Sensibilidade")

dat_text <- data.frame(
  Modelo    = c("Ridge","Lasso", "Elastic","Logit"),
  AUC       = round(calc_auc(roc)$AUC, 4),
  BASE      = c(rep("Teste",4),rep("Validação",4)),
  N.Vars    = c(vars_ridge[2],vars_lasso[2],vars_elastic[2], vars_logit[2])
)
dat_text0 <- dat_text %>% filter(BASE == aux) 
dat_text0 <- dat_text0[,c("Modelo","AUC","N.Vars")]
rownames(dat_text0) <- NULL

roc + geom_abline(intercept = 0, slope = 1, linetype = "dashed", size = 1) +
  scale_color_hue(l = 20, c = 100) +
  facet_grid(. ~ base ) +
  #  guides(colour = FALSE) +  
  annotation_custom(tableGrob(dat_text0, rows = NULL), xmin=0.75, xmax=0.85, ymin=0.10, ymax=0.20) +
  theme(legend.title=element_blank(),
        strip.background = element_rect(fill="black"),
        strip.text = element_text(size = 15, colour="white")) -> roc3
roc3

# ----------------------------------------------------------------------------------

roc0 <- grid.arrange(roc2, roc3, nrow = 1, widths = c(1.72, 2))
grid.draw(roc0)
ggsave("roc0.png", roc0, 
       width = 14, height = 7)      

ggauc <- grid.arrange(ggridge,gglasso,ggelastic,  nrow = 1)
grid.draw(ggauc)
ggsave("ggauc.png", ggauc, 
       width = 14, height = 4)      


