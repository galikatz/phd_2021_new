# install.packages(c("ggplot2", "ggpubr", "tidyverse", "broom", "AICcmodavg", "afex", "emmeans", "parameters", "effectsize", "ggthemes"))
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(broom)
library(rstatix)
library(dplyr)
library(afex)
library(emmeans)
library(effectsize)
library(ggthemes)

master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/architecture/architecture_anova.csv", header = TRUE,
                   colClasses = c("numeric", "factor", "factor", "factor", "factor", "numeric"))
#Index	UNIQUE_SUBJECT_UID	Task	Train	Test	Generations	Congruity	Ratio	Validation Accuracy
summary(master)

#########
# Optimizers
#########
group_by_df = master %>% group_by(Task,Train, Optimizer) %>%
                   summarise(Accuracy = mean(Val_Accuracy),
                             std = sd(Val_Accuracy),
                             .groups = 'drop')

# final_test_size_df <- subset(group_by_df, (Task == 'size'))
p_overall <- ggplot(group_by_df, aes(x=Optimizer, y=Accuracy)) +
  theme_classic2() +  scale_fill_manual(values = c("#80447B", "#CA91C4"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Optimizer, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Optimizers") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
p_overall + facet_grid(Train ~ Task)


#########
# Activation
########
group_by_df2 = master %>% group_by(Task,Train, Activation) %>%
                   summarise(Accuracy = mean(Val_Accuracy),
                             std = sd(Val_Accuracy),
                             .groups = 'drop')

# final_test_size_df <- subset(group_by_df, (Task == 'size'))
p_overall_activation <- ggplot(group_by_df2, aes(x=Activation, y=Accuracy)) +
  theme_classic2() +  scale_fill_manual(values = c("#80447B", "#CA91C4"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Activation, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Activation Functions") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
p_overall_activation + facet_grid(Train ~ Task)

