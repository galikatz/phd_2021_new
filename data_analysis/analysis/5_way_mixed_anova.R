# install.packages(c("ggplot2", "ggpubr", "tidyverse", "broom", "AICcmodavg", "afex", "emmeans"))
#install.packages(c("emmeans"))
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(broom)
library(rstatix)
library(dplyr)
library(afex)
library(emmeans)
master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/anova_df_for_R_new.csv", header = TRUE,
                      colClasses = c("numeric", "factor", "factor", "factor", "factor", "factor", "factor", "numeric"))
summary(master)


#this works:

(fit <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruency', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes')))


#emmeans
# emmeans(fit, pairwise ~ Task)

# emmip(fit,  Train ~ Test | Task)

# emmip(fit,  Congruency ~ Test | Task)

emmip(fit, Congruency  ~  Ratio | Task)

emmip(fit, Congruency  ~  Ratio | Test)

emmip(fit, Congruency  ~  Ratio | Task * Test)