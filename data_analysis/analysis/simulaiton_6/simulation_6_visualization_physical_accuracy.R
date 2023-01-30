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

master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/simulaiton_6/simulation_6_anova_df_for_R_15_01_23_physical_accuracy.csv", header = TRUE,
                   colClasses = c("numeric", "factor", "factor", "factor", "factor", "numeric", "factor", "factor", "numeric"))
#Index	UNIQUE_SUBJECT_UID	Task	Train	Test	Generations	Congruity	Ratio	Validation Accuracy
summary(master)

group_by_df = master %>% group_by(Task,Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')

##########################################
# Overall validation  - overall congruity
##########################################
group_by_df_overall_congruity = master %>% group_by(Task, Train) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')


# View(master_df_goup_by)
##############################
# Overall validation accuracy
##############################
p_overall <- ggplot(group_by_df, aes(x=Task, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#911A0E", "#D47970"))+ #scale_fill_brewer(palette="Set1") +
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
  theme_classic2() +  scale_fill_manual(values = c("#055905", "#9DD39D"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Size Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))

p_size + facet_grid(Train ~ Test)

#############
# Count-size
#############
count_size_df <- subset(master_df_goup_by, (Task == 'count-size'))
count_size_df$Train <- factor(count_size_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )

count_size_df$Test <- factor(count_size_df$Test, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Test AD-controlled", "Test TS-controlled", "Test CH-controlled")
                  )

p_count_size <- ggplot(count_size_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#DE821A", "#E8B071"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Count-size Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
p_count_size + facet_grid(Train ~ Test)

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
  theme_classic2() +  scale_fill_manual(values = c("#2980B9", "#85CCEE"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Colors-Count Task") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
#p_count
p_colors_count + facet_grid(Train ~ Test)


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
  theme_classic2() +  scale_fill_manual(values = c("#674099", "#D2B4DE")) +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Evolving to perceive size while controlling different physical properies and numerical ratios") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))

p_control_compare_with_ratio_df + facet_grid(Train ~ Task)