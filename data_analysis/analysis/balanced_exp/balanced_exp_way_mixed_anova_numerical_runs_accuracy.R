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
master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/balanced_exp/anova_df_for_R_06_02_23_numerical_accuracy.csv", header = TRUE,
                   colClasses = c("numeric", "factor", "factor", "factor", "factor", "numeric", "factor", "factor", "numeric"))
#Index	UNIQUE_SUBJECT_UID	Task	Train	Test	Generations	Congruity	Ratio	Validation Accuracy
summary(master)

master$Task <- factor(master$Task, levels = c("count", "size-count", "colors-count"),
                  labels = c("numerical", "physical-numerical", "colors-numerical")
                  )
#5 way anova - dependent variable Accuracy:
(fit <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes'), ))
(fit_nice <- aov_ez('UNIQUE_SUBJECT_UID','Validation.Accuracy',master,
                within = c('Test', 'Congruity', 'Ratio'),
                between= c('Task', 'Train'),
                anova_table = list(es = 'pes'),return="nice"))


# this is the entire 5-way interaction !!!!
five_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Test * Ratio * Congruity)
contrasts_five_way_interaction_not_split <- pairs(five_way_interaction_not_split_into_simple_cong_effects)
contrasts_five_way_interaction_table_not_split <- parameters::model_parameters(contrasts_five_way_interaction_not_split)


#########################################################################
# No added value when training with intrinsic but testing with extrinsic while
#########################################################################

cont1 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled AD.controlled X56 Incongruent) - (numerical AD-controlled CH.controlled X56 Incongruent)'))
# (numerical AD-controlled AD.controlled X56 Incongruent) - (numerical AD-controlled CH.controlled X56 Incongruent) |        0.50 | 0.03 | [0.43, 0.57] |  14.40 | < .001
t_to_eta2(t=14.40,df_error=170)=0.55

cont2 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled AD.controlled X63 Incongruent) - (numerical AD-controlled TS.controlled X63 Incongruent)'))
# (numerical AD-controlled AD.controlled X63 Incongruent) - (numerical AD-controlled TS.controlled X63 Incongruent) |        0.19 | 0.03 | [0.14, 0.25] |   6.97 | 0.023
t_to_eta2(t=6.97,df_error=170)=0.22

cont3 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled AD.controlled X71 Congruent) - (numerical AD-controlled TS.controlled X71 Congruent)'))
# (numerical AD-controlled AD.controlled X71 Congruent) - (numerical AD-controlled TS.controlled X71 Congruent) |        0.30 | 0.03 | [0.25, 0.36] |  11.19 | < .001
t_to_eta2(t=11.19,df_error=170)=0.42

cont4 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled AD.controlled X63 Congruent) - (numerical AD-controlled CH.controlled X63 Congruent)'))
# (numerical AD-controlled AD.controlled X63 Congruent) - (numerical AD-controlled CH.controlled X63 Congruent) |        0.12 | 0.02 | [0.09, 0.15] |   7.47 | 0.002
t_to_eta2(t=7.47,df_error=170)=0.25

#########################################################################
# There is an added value when training on extrinsic and testing with intinsic
#########################################################################
cont5 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled AD.controlled X50 Incongruent) - (numerical TS-controlled AD.controlled X50 Incongruent)')) - ns!

cont6 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled AD.controlled X63 Incongruent) - (numerical CH-controlled AD.controlled X63 Incongruent)')) - ns


#########################################################################
# Distance effect
#########################################################################
cont7 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled TS.controlled X50 Congruent) - (numerical AD-controlled TS.controlled X71 Congruent)'))
# (numerical AD-controlled TS.controlled X50 Congruent) - (numerical AD-controlled TS.controlled X71 Congruent) |        0.29 | 0.03 | [0.22, 0.35] |   9.32 | < .001
t_to_eta2(t=9.32,df_error=170)=0.34

cont8 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled TS.controlled X63 Congruent) - (numerical AD-controlled TS.controlled X86 Congruent)'))
# (numerical AD-controlled TS.controlled X63 Congruent) - (numerical AD-controlled TS.controlled X86 Congruent) |        0.37 | 0.02 | [0.32, 0.42] |  15.37 | < .001
t_to_eta2(t=15.37,df_error=170)=0.58

cont9 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled TS.controlled X50 Incongruent) - (numerical AD-controlled TS.controlled X71 Incongruent)'))
# (numerical AD-controlled TS.controlled X50 Incongruent) - (numerical AD-controlled TS.controlled X71 Incongruent) |        0.35 | 0.04 | [0.27, 0.44] |   8.29 | < .001
t_to_eta2(t=8.29,df_error=170)=0.29

cont10 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical AD-controlled TS.controlled X50 Incongruent) - (numerical AD-controlled TS.controlled X86 Incongruent)'))
# (numerical AD-controlled TS.controlled X50 Incongruent) - (numerical AD-controlled TS.controlled X86 Incongruent) |        0.37 | 0.04 | [0.29, 0.46] |   8.73 | < .001
t_to_eta2(t=8.73,df_error=170)=0.31

#########################################################################
# Ratio 50
#########################################################################
cont11 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical TS-controlled TS.controlled X50 Incongruent) - (numerical TS-controlled TS.controlled X56 Incongruent)'))
# (numerical TS-controlled TS.controlled X50 Incongruent) - (numerical TS-controlled TS.controlled X56 Incongruent) |       -0.40 | 0.04 | [-0.48, -0.32] |  -9.81 | < .001
t_to_eta2(t=-9.81,df_error=170)=0.36

cont12 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical AD-controlled AD.controlled X75 Incongruent) - (physical-numerical AD-controlled TS.controlled X75 Incongruent)'))
# (physical-numerical AD-controlled AD.controlled X75 Incongruent) - (physical-numerical AD-controlled TS.controlled X75 Incongruent) |        0.19 | 0.03 | [0.14, 0.24] |   7.30 | 0.005
t_to_eta2(t=7.30,df_error=170)=0.24

#########################################################################
# physical-numerical
#########################################################################
cont13 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical AD-controlled TS.controlled X50 Congruent) - (physical-numerical AD-controlled TS.controlled X71 Congruent)'))
# (physical-numerical AD-controlled TS.controlled X50 Congruent) - (physical-numerical AD-controlled TS.controlled X71 Congruent) |        0.30 | 0.03 | [0.23, 0.36] |   9.64 | < .001
t_to_eta2(t=9.64,df_error=170)=0.35

cont14 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical AD-controlled TS.controlled X63 Congruent) - (physical-numerical AD-controlled TS.controlled X86 Congruent)'))
# (physical-numerical AD-controlled TS.controlled X63 Congruent) - (physical-numerical AD-controlled TS.controlled X86 Congruent) |        0.29 | 0.02 | [0.24, 0.34] |  12.15 | < .001
t_to_eta2(t=12.15,df_error=170)=0.46

> cont15 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical AD-controlled TS.controlled X50 Incongruent) - (physical-numerical AD-controlled TS.controlled X86 Incongruent)'))
# (physical-numerical AD-controlled TS.controlled X50 Incongruent) - (physical-numerical AD-controlled TS.controlled X86 Incongruent) |        0.36 | 0.04 | [0.28, 0.45] |   8.50 | < .001
t_to_eta2(t=8.50,df_error=170)=0.30


#Training with CH and testing with CH vs trainin with CH and testing with AD ratio 50
cont16 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical CH-controlled AD.controlled X50 Incongruent) - (physical-numerical CH-controlled CH.controlled X50 Incongruent)'))



#########################################################################
# ratio 50
#########################################################################
cont16 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical CH-controlled CH.controlled X50 Incongruent) - (physical-numerical CH-controlled CH.controlled X56 Incongruent)'))
# (physical-numerical CH-controlled CH.controlled X50 Incongruent) - (physical-numerical CH-controlled CH.controlled X56 Incongruent) |       -0.28 | 0.04 | [-0.35, -0.21] |  -7.58 | 0.001
t_to_eta2(t=-7.58,df_error=170)=0.25


#########################################################################
# colors
#########################################################################
cont17 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-numerical TS-controlled TS.controlled X50 Incongruent) - (colors-numerical TS-controlled TS.controlled X71 Incongruent)'))
# (colors-numerical TS-controlled TS.controlled X50 Incongruent) - (colors-numerical TS-controlled TS.controlled X71 Incongruent) |        0.30 | 0.04 | [0.22, 0.39] |   7.12 | 0.012
t_to_eta2(t=7.12,df_error=170)=0.23

cont18 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(colors-numerical TS-controlled TS.controlled X63 Incongruent) - (colors-numerical TS-controlled TS.controlled X86 Incongruent)'))
# (colors-numerical TS-controlled TS.controlled X63 Incongruent) - (colors-numerical TS-controlled TS.controlled X86 Incongruent) |        0.21 | 0.02 | [0.17, 0.25] |  10.09 | < .001
t_to_eta2(t=10.09,df_error=170)=0.37

cont19 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical CH-controlled AD.controlled X86 Incongruent) - (colors-numerical CH-controlled AD.controlled X86 Incongruent)'))
# (physical-numerical CH-controlled AD.controlled X86 Incongruent) - (colors-numerical CH-controlled AD.controlled X86 Incongruent) |        0.27 | 0.04 | [0.20, 0.34] |   7.35 | 0.004
t_to_eta2(t=7.35,df_error=170)=0.24

cont20 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical TS-controlled TS.controlled X50 Incongruent) - (physical-numerical TS-controlled TS.controlled X50 Incongruent)'))
# (numerical TS-controlled TS.controlled X50 Incongruent) - (physical-numerical TS-controlled TS.controlled X50 Incongruent) |       -0.31 | 0.04 | [-0.40, -0.23] |  -7.51 | 0.002
t_to_eta2(t=-7.51,df_error=170)=0.25

cont21 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(numerical TS-controlled TS.controlled X50 Incongruent) - (colors-numerical TS-controlled TS.controlled X50 Incongruent)'))
# (numerical TS-controlled TS.controlled X50 Incongruent) - (colors-numerical TS-controlled TS.controlled X50 Incongruent) |       -0.51 | 0.04 | [-0.59, -0.43] | -12.33 | < .001
t_to_eta2(t=-12.33,df_error=170)=0.48

cont22 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical-numerical TS-controlled TS.controlled X50 Incongruent) - (colors-numerical TS-controlled TS.controlled X50 Incongruent)'))
# (numerical TS-controlled TS.controlled X50 Incongruent) - (colors-numerical TS-controlled TS.controlled X50 Incongruent) |       -0.51 | 0.04 | [-0.59, -0.43] | -12.33 | < .001
t_to_eta2(t=-12.33,df_error=170)=0.48


#######################
# three way interaction
#######################
three_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Congruity)
contrasts_three_way_interaction_not_split <- pairs(three_way_interaction_not_split_into_simple_cong_effects)
contrasts_three_way_interaction_table_not_split <- parameters::model_parameters(contrasts_three_way_interaction_not_split)



cont15 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(physical-numerical AD-controlled Incongruent) - (colors-numerical AD-controlled Incongruent)')) - ns
cont16 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(physical-numerical TS-controlled Incongruent) - (colors-numerical TS-controlled Incongruent)'))

cont16 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(physical-numerical TS-controlled Congruent) - (colors-numerical TS-controlled Congruent)'))

cont16 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(numerical TS-controlled Incongruent) - (physical-numerical TS-controlled Incongruent)'))
# (numerical TS-controlled Incongruent) - (physical-numerical TS-controlled Incongruent) |       -0.13 | 0.02 | [-0.17, -0.09] |  -5.95 | < .001



#######################
# four way interaction
#######################
four_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Test * Congruity)
contrasts_four_way_interaction_not_split <- pairs(four_way_interaction_not_split_into_simple_cong_effects)
contrasts_four_way_interaction_table_not_split <- parameters::model_parameters(contrasts_four_way_interaction_not_split)

cont13 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(colors-numerical AD-controlled CH.controlled Congruent) - (colors-numerical CH-controlled CH.controlled Congruent)'))
# (colors-numerical AD-controlled CH.controlled Congruent) - (colors-numerical CH-controlled CH.controlled Congruent) |        0.25 | 0.04 | [0.18, 0.33] |   6.57 | < .001
t_to_eta2(t=6.57,df_error=170)=0.20


cont15 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(colors-numerical CH-controlled CH.controlled Congruent) - (colors-numerical TS-controlled CH.controlled Congruent)'))
# (colors-numerical AD-controlled CH.controlled Congruent) - (colors-numerical CH-controlled CH.controlled Congruent) |        0.25 | 0.04 | [0.18, 0.33] |   6.57 | < .001
t_to_eta2(t=6.57,df_error=170)=0.20


