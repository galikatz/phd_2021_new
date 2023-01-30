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
p_overall <- ggplot(final_test_size_df, aes(x=Task, y=Accuracy, fill=Congruity)) +
  theme_classic2() +  scale_fill_manual(values = c("#80447B", "#CA91C4"))+ #scale_fill_brewer(palette="Set1") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  geom_errorbar( aes(x=Task, ymin=Accuracy-std, ymax=Accuracy+std),
                 width=0.3, colour="black", alpha=0.5, size=0.5, position=position_dodge(.9)) +
  ggtitle("Count-size accuracy final test size") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
p_overall

final_test_count_df <- subset(group_by_df, (Task == 'count'))
final_test_colors_df <- subset(group_by_df, (Task == 'colors'))

#########################
# Group by interactions
#########################
master_df_group_by = master %>% group_by(Task, Train, Congruity, Ratio) %>%
                   summarise(Accuracy = mean(Validation.Accuracy),
                             std = sd(Validation.Accuracy),
                             .groups = 'drop')

(fit <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes'), ))
(fit_nice <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes'),return="nice"))
#return="nice"
summary(fit)
###########################################
# this is the entire 4-way interaction !!!!
###########################################
four_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Ratio * Congruity)
contrasts_four_way_interaction_not_split <- pairs(four_way_interaction_not_split_into_simple_cong_effects)
contrasts_four_way_interaction_table_not_split <- parameters::model_parameters(contrasts_four_way_interaction_not_split)

cont1 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(size AD-controlled X56 Congruent) - (size AD-controlled X56 Incongruent)'))
# (size AD-controlled X56 Congruent) - (size AD-controlled X56 Incongruent) |        0.83 | 0.05 | [0.74, 0.92] |  18.39 | < .001
t_to_eta2(t=18.39,df_error=169)=0.67


cont2 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(size CH-controlled X63 Congruent) - (size CH-controlled X63 Incongruent)'))
# (size CH-controlled X63 Congruent) - (size CH-controlled X63 Incongruent) |        0.89 | 0.04 | [0.81, 0.98] |  20.10 | < .001
t_to_eta2(t=20.10 ,df_error=169)=0.71

cont3 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(size AD-controlled X50 Congruent) - (size AD-controlled X50 Incongruent)'))
# (size AD-controlled X50 Congruent) - (size AD-controlled X50 Incongruent) |       -0.93 | 0.05 | [-1.03, -0.83] | -18.89 | < .001
t_to_eta2(t=-18.89 ,df_error=169)=0.68

######################################
# three way interactions overall ratios
######################################

three_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Congruity)
contrasts_three_way_interaction_not_split <- pairs(three_way_interaction_not_split_into_simple_cong_effects)
contrasts_three_way_interaction_table_not_split <- parameters::model_parameters(contrasts_three_way_interaction_not_split)


cont4 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(count AD-controlled Congruent) - (count AD-controlled Incongruent)'))
# (count AD-controlled Congruent) - (count AD-controlled Incongruent) |        0.84 | 0.03 | [0.77, 0.91] |  23.98 | < .001
t_to_eta2(t=23.98 ,df_error=169)=0.77


cont5 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(count TS-controlled Congruent) - (count TS-controlled Incongruent)'))
# (count TS-controlled Congruent) - (count TS-controlled Incongruent) |        0.76 | 0.03 | [0.69, 0.83] |  21.85 | < .001
t_to_eta2(t=21.85 ,df_error=169)=0.74

cont6 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(count CH-controlled Congruent) - (count CH-controlled Incongruent)'))
# (count CH-controlled Congruent) - (count CH-controlled Incongruent) |        0.85 | 0.03 | [0.78, 0.92] |  24.28 | < .001
t_to_eta2(t=24.28 ,df_error=169)=0.78