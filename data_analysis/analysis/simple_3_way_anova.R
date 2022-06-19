# Title     : Anova - from size to counting
# Created by: gali.k
# Created on: 15/01/2022

# install.packages(c("ggplot2", "ggpubr", "tidyverse", "broom", "AICcmodavg"))
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(broom)
#library(AICcmodavg)

expriment_data <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/rawdata/aggregated/fortmatted_file_categorical.csv",
                      header = TRUE)
attach(expriment_data)
# Is_congruent<-factor(Is_congruent,c(0,1),labels=c('Cong','Incong'))
# Physical_property<-factor(Physical_property,c(1,2,3),labels=c('AD','TS','CH'))

summary(expriment_data)

three_way_anova_acc <- aov(Validation_accuracy ~ Physical_property_cat + Task + Is_congruent_cat, data = expriment_data)
print("############ three way anova ############")
summary(three_way_anova_acc)

# three_way_anova_plot <- ggplot(expriment_data, aes(x = Task, y = Validation_accuracy, group=Physical_property)) +
#   geom_point(cex = 2, pch = 1.0,position = position_jitter(w = 0.1, h = 0))

# three_way_anova_plot <- three_way_anova_plot +
#   geom_text(data=expriment_data, label=expriment_data$Physical_property, vjust = -8, size = 1) +
#   facet_wrap(~ Is_congruent)

# three_way_anova_plot <- three_way_anova_plot +
#   geom_text(data=expriment_data, label=expriment_data$Is_congruent, vjust = -8, size = 1) +
#   facet_wrap(~ Physical_property)
#
#
# three_way_anova_plot

#post hoc:
interaction <- aov(Validation_accuracy ~ Task * Is_congruent_cat * Physical_property_cat, data = expriment_data)
print("############ interactions ############")
summary(interaction)


# print("############ finding the best model ############")
# library(AICcmodavg)
#
# model.set <- list(three_way_anova_acc, interaction)
# model.names <- c("three_way_anova_acc", "interaction")
#
# aictab(model.set, modnames = model.names)

# print('############# check for homoscedasticity ############')
# par(mfrow=c(2,2))
# plot(three_way_anova_acc)
# par(mfrow=c(1,1))

print('###############  post hoc test ################')
tukey_three_way_anova_acc<-TukeyHSD(three_way_anova_acc)

tukey_three_way_anova_acc

# plot(aov(), las = 1)