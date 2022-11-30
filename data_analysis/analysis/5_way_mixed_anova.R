# install.packages(c("ggplot2", "ggpubr", "tidyverse", "broom", "AICcmodavg", "afex"))
#install.packages(c("afex"))
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(broom)
library(rstatix)
library(dplyr)
library(afex)
master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/anova_df_for_R_new.csv", header = TRUE,
                      colClasses = c("numeric", "factor", "factor", "factor", "factor", "factor", "factor", "numeric"))
summary(master)


#this works:

(fit <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruency', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes')))


#emmeans