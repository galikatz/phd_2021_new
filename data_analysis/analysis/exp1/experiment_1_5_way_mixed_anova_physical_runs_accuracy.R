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
#Index,UNIQUE_SUBJECT_UID,Task,Train,Test,Generations,Congruity,Ratio,Validation Accuracy
summary(master)


#5 way anova - dependent variable Accuracy:
(fit_nice <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes'), return="nice"))

(fit <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes')))
#return="nice"
summary(fit)

# mm_task_congruity <- emmeans(fit, ~ Task  * Congruity)
# # for getting contrasts
# contrasts_task_congruity<- pairs(mm_task_congruity)
# task_train_congruity_contrasts_table = parameters::model_parameters(contrasts_task_congruity)


#Five-way interaction
# this is the entire 5-way interaction !!!!
five_way_interaction <- emmeans(fit, ~ Task * Train * Test * Ratio * Congruity)
contrasts_five_way_interaction <- pairs(five_way_interaction)
contrasts_five_way_interaction_table <- parameters::model_parameters(contrasts_five_way_interaction)

#size contrasts:
cont1 <-subset(contrasts_five_way_interaction_table, (contrast == '(size CH-controlled AD.controlled X56 Incongruent) - (size CH-controlled AD.controlled X71 Incongruent)'))
#(size CH-controlled AD.controlled X56 Incongruent) - (size CH-controlled AD.controlled X71 Incongruent) |        0.21 | 0.02 | [0.17, 0.25] |  10.08 | < .001
t_to_eta2(t=10.08,df_error=169) = 0.38

cont2 <-subset(contrasts_five_way_interaction_table, (contrast == '(size CH-controlled AD.controlled X63 Incongruent) - (size CH-controlled AD.controlled X86 Incongruent)'))
# (size CH-controlled AD.controlled X63 Incongruent) - (size CH-controlled AD.controlled X86 Incongruent) |        0.15 | 0.02 | [0.11, 0.19] |   6.87 | 0.036
t_to_eta2(t=6.87,df_error=169) = 0.22

cont3 <-subset(contrasts_five_way_interaction_table, (contrast == '(size AD-controlled TS.controlled X50 Incongruent) - (size AD-controlled TS.controlled X86 Incongruent)'))
# (size AD-controlled TS.controlled X50 Incongruent) - (size AD-controlled TS.controlled X86 Incongruent) |        0.32 | 0.04 | [0.25, 0.40] |   9.00 | < .001
t_to_eta2(t=9,df_error=169) = 0.32

cont5 <-subset(contrasts_five_way_interaction_table, (contrast == '(size TS-controlled TS.controlled X86 Congruent) - (size TS-controlled TS.controlled X86 Incongruent)'))
# (size TS-controlled TS.controlled X86 Congruent) - (size TS-controlled TS.controlled X86 Incongruent) |       -0.25 | 0.04 | [-0.33, -0.17] |  -6.01 | 0.486
t_to_eta2(t=-6.01,df_error=169) = 0.32

#congruity effect
cont6 <-subset(contrasts_five_way_interaction_table, (contrast == '(size CH-controlled TS.controlled X71 Congruent) - (size CH-controlled TS.controlled X71 Incongruent)'))
# (size CH-controlled TS.controlled X71 Congruent) - (size CH-controlled TS.controlled X71 Incongruent) |        0.35 | 0.04 | [0.27, 0.42] |   9.59 | < .001
t_to_eta2(t=9.59,df_error=169) = 0.35

cont7 <-subset(contrasts_five_way_interaction_table, (contrast == '(count-size CH-controlled AD.controlled X56 Congruent) - (count-size CH-controlled AD.controlled X71 Incongruent)'))
# (count-size CH-controlled AD.controlled X56 Congruent) - (count-size CH-controlled AD.controlled X71 Incongruent) |        0.49 | 0.06 | [0.37, 0.61] |   8.23 | < .001
t_to_eta2(t=8.23,df_error=169)=0.29

cont8 <-subset(contrasts_five_way_interaction_table, (contrast == '(count-size CH-controlled AD.controlled X63 Congruent) - (count-size CH-controlled AD.controlled X86 Incongruent)'))
# (count-size CH-controlled AD.controlled X63 Congruent) - (count-size CH-controlled AD.controlled X86 Incongruent) |        0.47 | 0.06 | [0.35, 0.59] |   7.55 | 0.002
t_to_eta2(t=7.55,df_error=169)=0.25

cont9 <-subset(contrasts_five_way_interaction_table, (contrast == '(count-size AD-controlled TS.controlled X63 Congruent) - (count-size AD-controlled TS.controlled X86 Incongruent)'))
# (count-size AD-controlled TS.controlled X63 Congruent) - (count-size AD-controlled TS.controlled X86 Incongruent) |        0.22 | 0.03 | [0.16, 0.28] |   7.21 | 0.008
t_to_eta2(t=7.21,df_error=169)=0.24

cont10 <-subset(contrasts_five_way_interaction_table, (contrast == '(count-size CH-controlled TS.controlled X71 Congruent) - (count-size CH-controlled TS.controlled X71 Incongruent)'))
# (count-size CH-controlled TS.controlled X71 Congruent) - (count-size CH-controlled TS.controlled X71 Incongruent) |        0.27 | 0.04 | [0.20, 0.34] |   7.44 | 0.003
t_to_eta2(t=7.44,df_error=169)=0.25

cont11 <-subset(contrasts_five_way_interaction_table, (contrast == '(count-size CH-controlled TS.controlled X50 Congruent) - (count-size CH-controlled TS.controlled X50 Incongruent)'))
# (count-size CH-controlled TS.controlled X50 Congruent) - (count-size CH-controlled TS.controlled X50 Incongruent) |        0.24 | 0.03 | [0.19, 0.29] |   9.55 | < .001
t_to_eta2(t=9.55,df_error=169)=0.35

cont12 <-subset(contrasts_five_way_interaction_table, (contrast == '(count-size CH-controlled AD.controlled X63 Congruent) - (count-size CH-controlled AD.controlled X63 Incongruent)'))
# (count-size CH-controlled AD.controlled X63 Congruent) - (count-size CH-controlled AD.controlled X63 Incongruent) |        0.33 | 0.05 | [0.24, 0.43] |   6.85 | 0.039
t_to_eta2(t=6.85,df_error=169)=0.22

cont13 <-subset(contrasts_five_way_interaction_table, (contrast == '(count-size CH-controlled AD.controlled X71 Congruent) - (count-size CH-controlled AD.controlled X71 Incongruent)'))
# (count-size CH-controlled AD.controlled X71 Congruent) - (count-size CH-controlled AD.controlled X71 Incongruent) |        0.45 | 0.06 | [0.33, 0.57] |   7.50 | 0.002
t_to_eta2(t=7.50,df_error=169)=0.25


# comparing to control
four_way_interaction <- emmeans(fit, ~ Task * Train * Ratio * Congruity)
contrasts_four_way_interaction <- pairs(four_way_interaction)
contrasts_four_way_interaction_table <- parameters::model_parameters(contrasts_four_way_interaction)

#colors vs size
cont15 <-subset(contrasts_four_way_interaction_table, (contrast == '(colors CH-controlled X86 Incongruent) - (size CH-controlled X86 Incongruent)'))
# (colors CH-controlled X86 Incongruent) - (size CH-controlled X86 Incongruent) |        0.25 | 0.04 | [0.17, 0.33] |   6.32 | 0.015
t_to_eta2(t=6.32,df_error=169)=0.19

cont16 <-subset(contrasts_four_way_interaction_table, (contrast == '(colors CH-controlled X86 Incongruent) - (count-size CH-controlled X86 Incongruent)'))
# (colors CH-controlled X86 Incongruent) - (count-size CH-controlled X86 Incongruent) |        0.29 | 0.04 | [0.21, 0.37] |   7.19 | < .001
t_to_eta2(t=7.19,df_error=169)=0.23

cont17 <-subset(contrasts_four_way_interaction_table, (contrast == '(colors CH-controlled X75 Incongruent) - (count-size CH-controlled X75 Incongruent)'))
# (colors CH-controlled X75 Incongruent) - (count-size CH-controlled X75 Incongruent) |        0.25 | 0.04 | [0.18, 0.32] |   7.00 | < .001
t_to_eta2(t=7.00,df_error=169)=0.22

cont18 <-subset(contrasts_four_way_interaction_table, (contrast == '(colors CH-controlled X75 Incongruent) - (size CH-controlled X75 Incongruent)'))
# (colors CH-controlled X75 Incongruent) - (size CH-controlled X75 Incongruent) |        0.24 | 0.04 | [0.17, 0.31] |   6.61 | 0.004
t_to_eta2(t=6.61,df_error=169)=0.21

cont18 <-subset(contrasts_four_way_interaction_table, (contrast == '(colors CH-controlled X71 Incongruent) - (size CH-controlled X71 Incongruent)'))
# (colors CH-controlled X71 Incongruent) - (size CH-controlled X71 Incongruent) |        0.31 | 0.04 | [0.24, 0.38] |   8.46 | < .001
t_to_eta2(t=8.46,df_error=169)=0.3

cont18 <-subset(contrasts_four_way_interaction_table, (contrast == '(colors CH-controlled X71 Incongruent) - (count-size CH-controlled X71 Incongruent)'))
# (colors CH-controlled X71 Incongruent) - (count-size CH-controlled X71 Incongruent) |        0.28 | 0.04 | [0.21, 0.36] |   7.78 | < .001
t_to_eta2(t=7.78,df_error=169)=0.26

# three way onteraction accross ratios

three_way_interaction <- emmeans(fit, ~ Task * Train * Congruity)
contrasts_three_way_interaction <- pairs(three_way_interaction)
contrasts_three_way_interaction_table <- parameters::model_parameters(contrasts_three_way_interaction)

cont20 <-subset(contrasts_three_way_interaction_table, (contrast == '(colors CH-controlled Congruent) - (count-size CH-controlled Congruent)'))
# (colors CH-controlled Congruent) - (count-size CH-controlled Congruent) |        0.06 | 0.05 | [-0.03, 0.16] |   1.32 | > .999