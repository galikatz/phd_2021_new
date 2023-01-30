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

master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/Simulation7/simulation_7_anova_df_for_R_26_01_23.csv", header = TRUE,
                   colClasses = c("numeric", "factor", "factor", "factor", "factor", "numeric", "factor", "factor", "numeric"))
#Index	UNIQUE_SUBJECT_UID	Task	Train	Test	Generations	Congruity	Ratio	Validation Accuracy
summary(master)

group_by_df = master %>% group_by(Task,Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
# # View(master_df_goup_by)
# ##########################################
# # Overall validation  - overall congruity
# ##########################################
# group_by_df_overall_congruity = master %>% group_by(Task, Train) %>%
#                    summarise(Accuracy = mean(Validation.Accuracy),
#                              std = sd(Validation.Accuracy),
#                              .groups = 'drop')
#
#
#
# ##############################
# # Overall validation accuracy
# ##############################
final_test_size_df <- subset(group_by_df, (Task == 'size'))
final_test_count_df <- subset(group_by_df, (Task == 'count'))
final_test_colors_df <- subset(group_by_df, (Task == 'colors'))

#########################
# Group by interactions
#########################
master_df_group_by = master %>% group_by(Task, Train, Congruity, Ratio) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
master_df_group_by$Accuracy = signif(master_df_group_by$Accuracy, digits = 2)
#############
# Size-Count final test size
#############

size_df <- subset(master_df_group_by, (Task == 'size'))
size_df$Train <- factor(size_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )
p_size <- ggplot(size_df, aes(x=Ratio, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#188C97", "#98E1E8"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_text( aes(label = Accuracy), colour = "black", size = 3, vjust = 1.5, position = position_dodge(.9)) +
  geom_point()+geom_hline(yintercept=0.5,col=2) +
  geom_errorbar( aes(x=Ratio, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Accuracy of Size-Count CNNs in Size Perception Final Test") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))

p_size + facet_grid(Task ~ Train)

#############
# Count-size final test size
#############

three_way_master_df_group_by = master %>% group_by(Task, Train, Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
three_way_master_df_group_by$Accuracy = signif(three_way_master_df_group_by$Accuracy, digits = 2)


count_df <- subset(three_way_master_df_group_by, (Task == 'count'))
count_df$Train <- factor(count_df$Train, levels = c("AD-controlled", "TS-controlled", "CH-controlled"),
                  labels = c("Train AD-controlled", "Train TS-controlled", "Train CH-controlled")
                  )
p_count <- ggplot(count_df, aes(x=Train, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#894CE1", "#DBCAF2"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_text( aes(label = Accuracy), colour = "black", size = 5, vjust = 1.5, position = position_dodge(.9)) +
  geom_point()+geom_hline(yintercept=0.5,col=2) +
  geom_errorbar( aes(x=Train, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Accuracy of Count-Size CNNs in Numerical Final Test") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))

p_count

#######################################
# Colors-count final test size
#######################################
















two_way_master_df_group_by = master %>% group_by(Task, Congruity) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')
two_way_master_df_group_by$Accuracy = signif(two_way_master_df_group_by$Accuracy, digits = 2)

colors_df <- subset(two_way_master_df_group_by, (Task == 'colors'))

p_colors <- ggplot(colors_df, aes(x=Task, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#4CA2E1", "#8ABDE3"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_text( aes(label = Accuracy), colour = "black", size = 5, vjust = 1.5, position = position_dodge(.9)) +
  geom_point()+geom_hline(yintercept=0.5,col=2) +
  geom_errorbar( aes(x=Task, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Accuracy of Colors-Count CNNs in Colors Classification Final Test") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))

p_colors