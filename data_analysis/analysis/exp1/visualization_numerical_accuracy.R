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

master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/exp1/Experiment_1_numerical_final_test_cong_labels_and_controlled.csv", header = TRUE,
                   colClasses = c("numeric", "factor", "factor", "factor", "factor", "factor", "factor", "numeric", "factor", "factor", "factor"))


group_by_df = master %>% group_by(Task,Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
# View(master_df_goup_by)
##############################
# Overall validation accuracy
##############################
p_overall <- ggplot(group_by_df, aes(x=Task, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#45B39D", "#A3E4D7"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Task, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Overall Validation Accuracy in Numerical Runs") +  theme(plot.title = element_text(hjust = 0.5)) +
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
# Count
#############
count_df <- subset(master_df_goup_by, (Task == 'count'))
count_df$Train <- factor(count_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )

count_df$Test <- factor(count_df$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled")
                  )
p_count <- ggplot(count_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("steelblue", "#AED6F1"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Count Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_count + facet_grid(Train ~ Test)

#############
# Size-count
#############
size_count_df <- subset(master_df_goup_by, (Task == 'size-count'))
size_count_df$Train <- factor(size_count_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )

size_count_df$Test <- factor(size_count_df$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled")
                  )

p_size_count <- ggplot(size_count_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#C0392B", "#F1948A"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Size-Count Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_size_count + facet_grid(Train ~ Test)

#############
# Color-count
#############
colors_count_df <- subset(master_df_goup_by, (Task == 'colors-count'))

colors_count_df$Train <- factor(colors_count_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )

colors_count_df$Test <- factor(colors_count_df$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled")
                  )

p_colors_count <- ggplot(colors_count_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#8E44AD", "#C39BD3"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Colors-Count Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_colors_count + facet_grid(Train ~ Test)


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
  theme_classic2() +  scale_fill_manual(values = c("#2471A3", "#148F77"))+ #scale_fill_brewer(palette="Set1") +
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
  theme_classic2() +  scale_fill_manual(values = c("#EC7063", "#8E44AD")) +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Evolving to count while controlling different physical properies and numerical ratios") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_control_compare_with_ratio_df + facet_grid(Train ~ Task)

