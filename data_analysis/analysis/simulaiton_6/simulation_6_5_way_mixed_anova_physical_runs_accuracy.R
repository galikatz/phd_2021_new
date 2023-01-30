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


#5 way anova - dependent variable Accuracy:
(fit <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes'), ))
(fit_nice <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes'),return="nice"))
#return="nice"
summary(fit)
#emmeans
# emmeans(fit, pairwise ~ Task)
# emmip(fit,  Train ~ Test | Task)
# emmip(fit,  congruity ~ Test | Task)
# emmip(fit, congruity  ~  Ratio | Task)
# emmip(fit, congruity  ~  Ratio | Test)
# emmip(fit, congruity  ~  Ratio | Task * Train * Test)
# m2 <- emmeans(fit,  Task * Ratio | congruity)

## (1) Task * Train * Congruity interaction
#----------------------------
# for getting marginal means:
mm_task_train_congruity <- emmeans(fit, ~ Task * Train * Congruity)
# for getting contrasts
contrasts_task_train_congruity<- pairs(mm_task_train_congruity)
task_train_congruity_contrasts_table = parameters::model_parameters(contrasts_task_train_congruity)

##########################################
# Differences compared to the control task
##########################################
# |   Congruity | Coefficient |   SE |         95% CI | t(170) |      p
#(colors-count Total Surface Area) - count Total Surface Area        | Incongruent |       -0.40 | 0.01 | [-0.43, -0.37] | -28.61 | < .001
#t_to_eta2(t=28.61,df_error=170) = 0.83

#size-count Convex Hull) - (size-count Total Surface Area)          | Incongruent |       -0.15 | 0.01 | [-0.18, -0.12] | -10.75 | < .001
#t_to_eta2(t=10.75,df_error=170) = 0.4

# cont30 <-subset(task_train_congruity_contrasts_table, (contrast == '(colors-count TS-controlled Congruent) - (count TS-controlled Congruent)'))
# (colors-count TS-controlled Congruent) - (count TS-controlled Congruent) |       -0.40 | 0.02 | [-0.43, -0.37] | -25.69 | < .001
# t_to_eta2(t=-25.69,df_error=170)=0.80

# cont32 <-subset(task_train_congruity_contrasts_table, (contrast == '(colors-count TS-controlled Incongruent) - (size-count TS-controlled Incongruent)'))
# (colors-count TS-controlled Incongruent) - (size-count TS-controlled Incongruent) |       -0.42 | 0.01 | [-0.45, -0.39] | -30.10 | < .001
# t_to_eta2(t=-30.10,df_error=170)=0.84

# cont33 <-subset(task_train_congruity_contrasts_table, (contrast == '(colors-count TS-controlled Congruent) - (size-count TS-controlled Congruent)'))
# (colors-count TS-controlled Congruent) - (size-count TS-controlled Congruent) |       -0.42 | 0.02 | [-0.45, -0.39] | -27.56 | < .001
# t_to_eta2(t=-27.56,df_error=170)=0.82

##########################################
# Differences between count and size-count
##########################################




# plotting
#p1 <- afex_plot(fit, x = "Task", trace = "Train", panel = "Congruity", mapping = c("color", "fill"))
#p1 + theme(text = element_text(size = 20)) + ggtitle("Task and Train Interaction split by Congruity") + theme(plot.title = element_text(hjust = 0.5))
emmip(fit,  Congruity ~ Task | Train,
      xlab="Task",
      ylab="Validation Accuracy"
) +  theme_bw() +
  ggtitle("Task and Train Interaction") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20))

## Train * Test interaction
#----------------------------
# for getting marginal means:
mm_train_test_by_congruity <- emmeans(fit, ~ Train * Test | Congruity)
contrasts_train_test_by_congruity<- pairs(mm_train_test_by_congruity)
contrasts_train_test_table <- parameters::model_parameters(contrasts_train_test_by_congruity)
emmip(fit,  Congruity ~ Train | Test,
      xlab="Physical Property Controlled during Training",
      ylab="Validation Accuracy",
) + theme_bw() +
  ggtitle("Train and Test Interaction") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20))
#
# emmip(fit,  Congruity ~ Train | Test,
#       xlab="Physical Property Controlled during Training",
#       ylab="Validation Accuracy",
#       ) +
#   geom_bar(stat="identity", fill="white", position="dodge") + theme_bw() +
#   ggtitle("Train and Test Interaction") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   theme(text = element_text(size = 20))



# this is the entire 5-way interaction !!!!
five_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Test * Ratio * Congruity)
contrasts_five_way_interaction_not_split <- pairs(five_way_interaction_not_split_into_simple_cong_effects)
contrasts_five_way_interaction_table_not_split <- parameters::model_parameters(contrasts_five_way_interaction_not_split)

cont1 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count CH-controlled CH.controlled X75 Incongruent) - (size-count CH-controlled TS.controlled X75 Incongruent)'))
# (size-count CH-controlled CH.controlled X75 Incongruent) - (size-count CH-controlled TS.controlled X75 Incongruent) |        0.65 | 0.05 | [0.55, 0.74] |  13.39 | < .001
t_to_eta2(t=13.39,df_error=166)=0.52

cont2 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled AD.controlled X75 Congruent) - (count AD-controlled TS.controlled X75 Congruent)'))
# (count AD-controlled AD.controlled X75 Congruent) - (count AD-controlled TS.controlled X75 Congruent) |        0.35 | 0.04 | [0.28, 0.42] |   9.69 | < .001
t_to_eta2(t=9.69,df_error=166)=0.36

################################
# Ratio 50 exceptional behavior
################################
cont3 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled AD.controlled X50 Incongruent) - (count AD-controlled TS.controlled X50 Incongruent)'))
# (count AD-controlled AD.controlled X50 Incongruent) - (count AD-controlled TS.controlled X50 Incongruent) |       -0.50 | 0.05 | [-0.59, -0.41] | -10.58 | < .001
t_to_eta2(t=-10.58,df_error=166)=0.4

(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled AD.controlled X50 Incongruent) - (count AD-controlled AD.controlled X56 Incongruent)'))
# (count AD-controlled AD.controlled X50 Incongruent) - (count AD-controlled AD.controlled X56 Incongruent) |       -0.50 | 0.06 | [-0.61, -0.39] |  -8.83 | < .001
t_to_eta2(t=-8.83,df_error=166)=0.32


##########################
# Inverse congruity effect
##########################
cont5 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled TS.controlled X50 Congruent) - (count AD-controlled TS.controlled X50 Incongruent)'))
# (count AD-controlled TS.controlled X50 Congruent) - (count AD-controlled TS.controlled X50 Incongruent) |       -0.25 | 0.02 | [-0.29, -0.21] | -12.81 | < .001
t_to_eta2(t=-12.81,df_error=166)=0.5

cont6 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count CH-controlled TS.controlled X50 Congruent) - (count CH-controlled TS.controlled X50 Incongruent)'))
# (count CH-controlled TS.controlled X50 Congruent) - (count CH-controlled TS.controlled X50 Incongruent) |       -0.20 | 0.02 | [-0.24, -0.16] |  -9.99 | < .001
t_to_eta2(t=-9.99,df_error=166)=0.38

cont7 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-count AD-controlled AD.controlled X56 Congruent) - (colors-count AD-controlled AD.controlled X56 Incongruent)'))
# (colors-count AD-controlled AD.controlled X56 Congruent) - (colors-count AD-controlled AD.controlled X56 Incongruent) |       -0.17 | 0.01 | [-0.20, -0.14] | -12.13 | < .001
t_to_eta2(t=-12.13,df_error=166)=0.47

cont7 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-count AD-controlled AD.controlled X75 Congruent) - (colors-count AD-controlled AD.controlled X75 Incongruent)'))
# (colors-count AD-controlled AD.controlled X75 Congruent) - (colors-count AD-controlled AD.controlled X75 Incongruent) |       -0.08 | 8.40e-03 | [-0.10, -0.07] | -10.05 | < .001)
t_to_eta2(t=-10.05,df_error=166)=0.38


cont10 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-count AD-controlled TS.controlled X86 Congruent) - (colors-count AD-controlled TS.controlled X86 Incongruent)'))
# (colors-count AD-controlled TS.controlled X86 Congruent) - (colors-count AD-controlled TS.controlled X86 Incongruent) |       -0.13 | 0.02 | [-0.16, -0.10] |  -8.32 | < .001
t_to_eta2(t=-8.32,df_error=166)=0.29

cont11 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-count CH-controlled TS.controlled X86 Congruent) - (colors-count CH-controlled TS.controlled X86 Incongruent)'))
# (colors-count CH-controlled TS.controlled X86 Congruent) - (colors-count CH-controlled TS.controlled X86 Incongruent) |       -0.15 | 0.01 | [-0.18, -0.12] | -10.63 | < .001
t_to_eta2(t=-10.63,df_error=166)=0.41

##################
# distance effect
##################
cont8 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-count AD-controlled TS.controlled X50 Incongruent) - (colors-count AD-controlled TS.controlled X75 Incongruent)'))
# (colors-count AD-controlled TS.controlled X50 Incongruent) - (colors-count AD-controlled TS.controlled X75 Incongruent) |        0.43 | 0.03 | [0.37, 0.50] |  13.23 | < .001
t_to_eta2(t=13.23,df_error=166)=0.51

cont8 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-count AD-controlled TS.controlled X56 Incongruent) - (colors-count AD-controlled TS.controlled X71 Incongruent)'))
---------------------------------------------------------------------------
# (colors-count AD-controlled TS.controlled X56 Incongruent) - (colors-count AD-controlled TS.controlled X71 Incongruent) |        0.13 | 0.01 | [0.10, 0.15] |   9.46 | < .001
t_to_eta2(t=9.46,df_error=166)=0.35


#exceptionals:
cont9 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count AD-controlled TS.controlled X56 Incongruent) - (size-count AD-controlled TS.controlled X86 Incongruent)'))
# (size-count AD-controlled TS.controlled X56 Incongruent) - (size-count AD-controlled TS.controlled X86 Incongruent) |       -0.12 | 0.02 | [-0.15, -0.09] |  -7.82 | < .001
t_to_eta2(t=-7.82,df_error=166)=0.27

cont11 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-count CH-controlled TS.controlled X71 Congruent) - (colors-count CH-controlled TS.controlled X75 Congruent)'))
# (colors-count CH-controlled TS.controlled X71 Congruent) - (colors-count CH-controlled TS.controlled X75 Congruent) |       -0.24 | 0.01 | [-0.27, -0.21] | -16.30 | < .001
t_to_eta2(t=-16.30,df_error=166)=0.62


#######################
# three way interaction
#######################
three_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Congruity)
contrasts_three_way_interaction_not_split <- pairs(three_way_interaction_not_split_into_simple_cong_effects)
contrasts_three_way_interaction_table_not_split <- parameters::model_parameters(contrasts_three_way_interaction_not_split)

cont12 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(count AD-controlled Congruent) - (count TS-controlled Congruent)'))
# (count AD-controlled Congruent) - (count TS-controlled Congruent) |       -0.16 | 0.02 | [-0.20, -0.12] |  -8.12 | < .001
t_to_eta2(t=-8.12,df_error=166)=0.28

cont13 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(count AD-controlled Incongruent) - (count TS-controlled Incongruent)'))
# (count AD-controlled Incongruent) - (count TS-controlled Incongruent) |       -0.18 | 0.02 | [-0.21, -0.15] | -11.80 | < .001
t_to_eta2(t=-11.8,df_error=166)=0.46

cont15 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(size-count CH-controlled Congruent) - (size-count TS-controlled Congruent)'))
# (size-count CH-controlled Congruent) - (size-count TS-controlled Congruent) |       -0.11 | 0.02 | [-0.15, -0.07] |  -5.56 | < .001
t_to_eta2(t=-5.56,df_error=166)=0.16

cont14 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(size-count CH-controlled Incongruent) - (size-count TS-controlled Incongruent)'))
# (size-count CH-controlled Incongruent) - (size-count TS-controlled Incongruent) |       -0.13 | 0.02 | [-0.16, -0.10] |  -8.61 | < .001
t_to_eta2(t=-8.61,df_error=166)=0.31





#######################
# four way interaction
#######################
four_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Ratio * Congruity)
contrasts_four_way_interaction_not_split <- pairs(four_way_interaction_not_split_into_simple_cong_effects)
contrasts_four_way_interaction_table_not_split <- parameters::model_parameters(contrasts_four_way_interaction_not_split)


