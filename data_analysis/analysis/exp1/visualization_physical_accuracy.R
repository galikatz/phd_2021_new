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

master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/exp1/anova_df_for_R_30_12_22_physical.csv", header = TRUE,
                   colClasses = c("numeric", "factor","factor","factor","factor","numeric","factor","factor","numeric"))

group_by_df = master %>% group_by(Task,Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
# View(master_df_goup_by)
##############################
# Overall validation accuracy
##############################
p_overall <- ggplot(group_by_df, aes(x=Task, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#7D3C98", "#D2B4DE"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Task, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Overall Validation Accuracy in Physical Runs") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
p_overall

#########################
# Group by interactions
#########################
master_df_goup_by = master %>% group_by(Task, Train, Congruity, Test, Ratio) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
#############
# Size
#############
size_df <- subset(master_df_goup_by, (Task == 'size'))
size_df$Train <- factor(size_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )

size_df$Test <- factor(size_df$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled")
                  )
p_size <- ggplot(size_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#F88A07", "#F8D407"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Size Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_size
p_size + facet_grid(Train ~ Test)

#############
# count-size
#############
count_size_df <- subset(master_df_goup_by, (Task == 'count-size'))
count_size_df$Train <- factor(count_size_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )

count_size_df$Test <- factor(count_size_df$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled")
                  )

p_count_size <- ggplot(count_size_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#D21DA4", "#F0A7DD"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Count-Size Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_count_size + facet_grid(Train ~ Test)

#############
# Color
#############
colors_df <- subset(master_df_goup_by, (Task == 'colors'))

colors_df$Train <- factor(colors_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )

colors_df$Test <- factor(colors_df$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled")
                  )

p_colors <- ggplot(colors_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#649EA0", "#AAEDF0"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Colors Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))

p_colors + facet_grid(Train ~ Test)


#################
# distance effect
#################

master_df_goup_by$Train <- factor(master_df_goup_by$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled"))

master_df_goup_by$Test <- factor(master_df_goup_by$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled"))

distance_effect_df <- subset(master_df_goup_by, ((Task == 'count' | Task == 'size-count') & (Train == 'Train AD-controlled' | Train == 'Train CH-controlled') & Test == 'Test TS-controlled'))

View(distance_effect_df)

p_distance_effect <- ggplot(distance_effect_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#D35400", "#F0B27A"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Accuracy Decreases while Numrical Ratio Increases") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_distance_effect + facet_grid(Train ~ Test * Task)


#######################################################
# differences compared to control task - overall ratios
#######################################################
control_compare_group_by = master %>% group_by(Task, Train, Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
control_compare_group_by$Train <- factor(control_compare_group_by$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled"))


p_control_compare_df <- ggplot(control_compare_group_by, aes(x=Task, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#8876EB", "#C6BFF0"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Task, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Evolving on different physical properies") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_control_compare_df + facet_grid(~Train)

####################################################
# differences compared to control task - with ratio
####################################################
control_compare_group_by_with_ratio = master %>% group_by(Task, Train, Ratio, Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
control_compare_group_by_with_ratio$Train <- factor(control_compare_group_by_with_ratio$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled"))

p_control_compare_with_ratio_df <- ggplot(control_compare_group_by_with_ratio, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#168BA8", "#C85182")) +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Evolving to perceive size while controlling different physical properies and numerical ratios") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))

p_control_compare_with_ratio_df + facet_grid(Train ~ Task)

