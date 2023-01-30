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
#Index	UNIQUE_SUBJECT_UID	Task	Train_val	Test_val	Congruity_val	Ratio	Validation Accuracy	Congruity	Train	Test
master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/exp1/Experiment_1_numerical_final_test_cong_labels_and_controlled.csv", header = TRUE,
                   colClasses = c("numeric", "factor", "factor", "factor", "factor", "factor", "factor", "numeric", "factor", "factor", "factor"))
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


#Five-way interaction
#Task, Train, Test, Congruity and Ratio
five_way_interaction <- emmeans(fit, ~ Task * Train * Test * Ratio | Congruity)
contrasts_five_way_interaction<- pairs(five_way_interaction)
contrasts_five_way_interaction_table <- parameters::model_parameters(contrasts_five_way_interaction)
emmip(fit, Congruity  ~  Ratio | Task * Train * Test,
      xlab="Ratio",
      ylab="Validation Accuracy",
) + theme_bw() +
  ggtitle("Train and Test Interaction") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20))

contrasts_five_way_congruent <- subset(contrasts_five_way_interaction_table, (Congruity == 'Congruent'))
contrasts_five_way_incongruent <- subset(contrasts_five_way_interaction_table, (Congruity == 'Incongruent'))

#selected 5-way contrasts
#cont1 <-subset(contrasts_five_way_incongruent, (contrast == '(size-count AD-controlled CH.controlled X63) - (size-count AD-controlled TS.controlled X63)'))
# (size-count AD-controlled CH.controlled X63 Incongruent) - (size-count AD-controlled TS.controlled X63 Incongruent) |        0.41 | 0.02 | [0.37, 0.45] |  18.18 | < .001
# t_to_eta2(t=18.18,df_error=170) = 0.66
# cont2 <-subset(contrasts_five_way_incongruent, (contrast == '(size-count AD-controlled CH.controlled X50) - (size-count AD-controlled TS.controlled X50)'))
# (size-count AD-controlled CH.controlled X50 Incongruent) - (size-count AD-controlled TS.controlled X50 Incongruent) |       -0.67 | 0.03 | [-0.73, -0.61] | -20.77 | < .001
# t_to_eta2(t=-20.77,df_error=170) = 0.72

# this is the entire 5-way interaction !!!!
five_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Test * Ratio * Congruity)
contrasts_five_way_interaction_not_split <- pairs(five_way_interaction_not_split_into_simple_cong_effects)
contrasts_five_way_interaction_table_not_split <- parameters::model_parameters(contrasts_five_way_interaction_not_split)

# cont3 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count AD-controlled CH.controlled X50 Congruent) - (size-count AD-controlled TS.controlled X50 Incongruent)'))
#(size-count AD-controlled CH.controlled X50 Congruent) - (size-count AD-controlled TS.controlled X50 Incongruent) |       -0.52 | 0.03 | [-0.58, -0.47] | -18.22 | < .001
#  t_to_eta2(t=-18.22,df_error=170) = 0.66

# cont4 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count TS-controlled TS.controlled X75 Congruent) - (count TS-controlled TS.controlled X75 Incongruent)'))
# (count TS-controlled TS.controlled X75 Congruent) - (count TS-controlled TS.controlled X75 Incongruent) |       -0.12 | 7.49e-03 | [-0.14, -0.11] | -16.68 | < .001
# t_to_eta2(t=-16.68,df_error=170) = 0.62


# cont5 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled AD.controlled X75 Congruent) - (count AD-controlled AD.controlled X75 Incongruent)'))
# (count AD-controlled AD.controlled X75 Congruent) - (count AD-controlled AD.controlled X75 Incongruent) |       -0.07 | 8.79e-03 | [-0.09, -0.06] |  -8.53 | < .001
# t_to_eta2(t=-8.53,df_error=170) = 0.30


#####################################
#size-count distnce effects contrasts
#####################################
# CH [63 vs 86]
# cont7 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count CH-controlled TS.controlled X63 Incongruent) - (size-count CH-controlled TS.controlled X86 Incongruent)'))
# (size-count CH-controlled TS.controlled X63 Incongruent) - (size-count CH-controlled TS.controlled X86 Incongruent) |        0.13 | 7.65e-03 | [0.12, 0.15] |  17.54 | < .001
#  t_to_eta2(t=17.54,df_error=170) = 0.64

# CH [63 vs 75]
# cont17 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count CH-controlled TS.controlled X63 Incongruent) - (size-count CH-controlled TS.controlled X75 Incongruent)'))
# (size-count CH-controlled TS.controlled X63 Incongruent) - (size-count CH-controlled TS.controlled X75 Incongruent) |        0.18 | 8.85e-03 | [0.16, 0.20] |  20.23 | < .001
# t_to_eta2(t=20.23,df_error=170)=0.71

#AD [50 vs 71]
# cont12 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count AD-controlled TS.controlled X50 Incongruent) - (size-count AD-controlled TS.controlled X71 Incongruent)'))
# (size-count AD-controlled TS.controlled X50 Incongruent) - (size-count AD-controlled TS.controlled X71 Incongruent) |        0.34 | 0.02 | [0.29, 0.39] |  14.77 | < .001
#  t_to_eta2(t=14.77,df_error=170)=0.56

#AD [63 vs 75]
# 19
# (size-count AD-controlled TS.controlled X63 Incongruent) - (size-count AD-controlled TS.controlled X75 Incongruent) |        0.20 | 8.62e-03 | [0.19, 0.22] |  23.49 | < .001
# t_to_eta2(t=23.49,df_error=170)=0.76


#####################################
#count distnce effects contrasts
#####################################

# CH [63 VS 86]
# cont8 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count CH-controlled TS.controlled X63 Incongruent) - (count CH-controlled TS.controlled X86 Incongruent)'))
# (count CH-controlled TS.controlled X63 Incongruent) - (count CH-controlled TS.controlled X86 Incongruent) |        0.15 | 7.46e-03 | [0.14, 0.16] |  20.12 | < .001
# t_to_eta2(t=20.12,df_error=170) = 0.70

# CH [50 VS 71]
# cont15 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count CH-controlled TS.controlled X50 Incongruent) - (count CH-controlled TS.controlled X71 Incongruent)'))
# (count CH-controlled TS.controlled X50 Incongruent) - (count CH-controlled TS.controlled X71 Incongruent) |        0.35 | 0.02 | [0.30, 0.39] |  15.10 | < .001
# t_to_eta2(t=15.09,df_error=170)=0.57

#CH [63 vs 75]
# cont16 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count CH-controlled TS.controlled X63 Incongruent) - (count CH-controlled TS.controlled X75 Incongruent)'))
# (count CH-controlled TS.controlled X63 Incongruent) - (count CH-controlled TS.controlled X75 Incongruent) |        0.20 | 8.62e-03 | [0.18, 0.22] |  23.20 | < .001
# t_to_eta2(t=23.20,df_error=170)=0.76

# AD [63 V 75]
# cont9 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled TS.controlled X63 Incongruent) - (count AD-controlled TS.controlled X75 Incongruent)'))
# (count AD-controlled TS.controlled X63 Incongruent) - (count AD-controlled TS.controlled X75 Incongruent) |        0.19 | 8.62e-03 | [0.17, 0.20] |  21.75 | < .001
# t_to_eta2(t=21.75,df_error=170)=0.74

# AD  [50 VS 71]
# cont14 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled TS.controlled X50 Incongruent) - (count AD-controlled TS.controlled X71 Incongruent)'))
# (count AD-controlled TS.controlled X50 Incongruent) - (count AD-controlled TS.controlled X71 Incongruent) |        0.33 | 0.02 | [0.28, 0.38] |  14.34 | < .001
# t_to_eta2(t=14.34,df_error=170)=0.55

###################
#Congruity effects:
###################
# cont20 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count AD-controlled CH.controlled X50 Congruent) - (size-count AD-controlled CH.controlled X50 Incongruent)'))
# (size-count AD-controlled CH.controlled X50 Congruent) - (size-count AD-controlled CH.controlled X50 Incongruent) |        0.15 | 9.81e-03 | [0.13, 0.17] |  15.03 | < .001
# t_to_eta2(t=15.03,df_error=170)=0.57

# cont21 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count TS-controlled CH.controlled X50 Congruent) - (size-count TS-controlled CH.controlled X50 Incongruent)'))
# (size-count TS-controlled CH.controlled X50 Congruent) - (size-count TS-controlled CH.controlled X50 Incongruent) |        0.15 | 9.81e-03 | [0.13, 0.17] |  15.28 | < .001
# t_to_eta2(t=15.28,df_error=170)=0.58


# cont22 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(size-count TS-controlled TS.controlled X75 Congruent) - (size-count TS-controlled TS.controlled X75 Incongruent)'))
# (size-count TS-controlled TS.controlled X75 Congruent) - (size-count TS-controlled TS.controlled X75 Incongruent) |       -0.15 | 7.49e-03 | [-0.16, -0.14] | -20.02 | < .001
# t_to_eta2(t=15.28,df_error=170)=0.58

# cont22 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count TS-controlled CH.controlled X50 Congruent) - (count TS-controlled CH.controlled X50 Incongruent)'))
# (count TS-controlled CH.controlled X50 Congruent) - (count TS-controlled CH.controlled X50 Incongruent) |        0.13 | 9.81e-03 | [0.11, 0.15] |  13.50 | < .001
# t_to_eta2(t=13.50,df_error=170)=0.52

# cont23 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(count AD-controlled CH.controlled X50 Congruent) - (count AD-controlled CH.controlled X50 Incongruent)'))
# (count AD-controlled CH.controlled X50 Congruent) - (count AD-controlled CH.controlled X50 Incongruent) |        0.14 | 9.81e-03 | [0.12, 0.15] |  13.76 | < .001
# t_to_eta2(t=13.76,df_error=170)=0.53

#######################
# four way interaction
#######################
four_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Ratio * Congruity)
contrasts_four_way_interaction_not_split <- pairs(four_way_interaction_not_split_into_simple_cong_effects)
contrasts_four_way_interaction_table_not_split <- parameters::model_parameters(contrasts_four_way_interaction_not_split)

###########
# contrasts
###########
cont1 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(colors-count TS-controlled X50 Incongruent) - (count TS-controlled X50 Incongruent)'))
# (colors-count TS-controlled X50 Incongruent) - (count TS-controlled X50 Incongruent) |        0.62 | 0.03 | [0.57, 0.67] |  23.64 | < .001
t_to_eta2(23.64, df_error=170)=0.77

cont2 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(colors-count TS-controlled X50 Incongruent) - (size-count TS-controlled X50 Incongruent)'))
# (colors-count TS-controlled X50 Incongruent) - (size-count TS-controlled X50 Incongruent) |        0.66 | 0.03 | [0.61, 0.71] |  25.37 | < .001
t_to_eta2(25.37, df_error=170)=0.79

cont3 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(colors-count TS-controlled X75 Incongruent) - (count TS-controlled X75 Incongruent)'))
# (colors-count TS-controlled X75 Incongruent) - (count TS-controlled X75 Incongruent) |       -0.61 | 0.02 | [-0.65, -0.56] | -26.40 | < .001
t_to_eta2(-26.40, df_error=170)=0.80

cont4 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(colors-count TS-controlled X75 Incongruent) - (size-count TS-controlled X75 Incongruent)'))
# (colors-count TS-controlled X75 Incongruent) - (size-count TS-controlled X75 Incongruent) |       -0.65 | 0.02 | [-0.70, -0.60] | -28.17 | < .001
t_to_eta2(-28.17, df_error=170)=0.82

cont5 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(size-count CH-controlled X86 Incongruent) - (size-count TS-controlled X86 Incongruent)'))
# (size-count CH-controlled X86 Incongruent) - (size-count TS-controlled X86 Incongruent) |       -0.23 | 0.02 | [-0.28, -0.18] |  -9.38 | < .001
t_to_eta2(-9.38, df_error=170)=0.34

cont6 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(size-count AD-controlled X86 Incongruent) - (size-count TS-controlled X86 Incongruent)'))
# (size-count AD-controlled X86 Incongruent) - (size-count TS-controlled X86 Incongruent) |       -0.22 | 0.02 | [-0.26, -0.17] |  -8.82 | < .001
t_to_eta2(-8.82, df_error=170)=0.31

cont7 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(count AD-controlled X86 Incongruent) - (count TS-controlled X86 Incongruent)'))
# (count AD-controlled X86 Incongruent) - (count TS-controlled X86 Incongruent) |       -0.18 | 0.02 | [-0.23, -0.13] |  -7.46 | < .001
t_to_eta2(-7.46, df_error=170)=0.25

cont8 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(count CH-controlled X86 Incongruent) - (count TS-controlled X86 Incongruent)'))
# (count CH-controlled X86 Incongruent) - (count TS-controlled X86 Incongruent) |       -0.16 | 0.02 | [-0.21, -0.11] |  -6.58 | 0.005
t_to_eta2(-6.58, df_error=170)=0.20