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
library(tidyverse)


master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/balanced_exp/anova_df_for_R_06_02_23_physical_accuracy.csv", header = TRUE,
                   colClasses = c("numeric", "factor", "factor", "factor", "factor", "numeric", "factor", "factor", "numeric"))
#Index	UNIQUE_SUBJECT_UID	Task	Train	Test	Generations	Congruity	Ratio	Validation Accuracy
summary(master)

master$Task <- factor(master$Task, levels = c("colors", "size", "count-size"),
                  labels = c("colors", "physical", "numerical-physical")
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

sigificant_effects <- subset(contrasts_five_way_interaction_table_not_split, (p<=0.05))
# filter sigificant_effects for contrast containing the string 'numerical-physical AD-controlled TS.controlled'
filter1 <- subset(sigificant_effects, grepl('numerical-physical AD-controlled TS.controlled', contrast))
sigificant_effects_numerical_physical_AD_controlled_TS_controlled <- subset(filter1, grepl('numerical-physical TS-controlled TS.controlled', contrast))


filter2 <- subset(sigificant_effects, grepl('numerical-physical TS-controlled TS.controlled', contrast))
sigificant_effects_numerical_physical_CH_controlled_TS_controlled <- subset(filter2, grepl('numerical-physical AD-controlled TS.controlled', contrast))

#########################################################################
# No added value when training with intrinsic but testing with extrinsic while
#########################################################################

cont1 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical CH-controlled TS.controlled X50 Congruent) - (physical CH-controlled TS.controlled X50 Incongruent)'))
# (physical CH-controlled TS.controlled X50 Congruent) - (physical CH-controlled TS.controlled X50 Incongruent) |        0.24 | 0.03 | [0.17, 0.31] |   6.93 | 0.028
t_to_eta2(t=6.93,df_error=168)=.22


cont2 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical CH-controlled AD.controlled X50 Congruent) - (physical CH-controlled AD.controlled X50 Incongruent)'))
# (physical CH-controlled AD.controlled X56 Congruent) - (physical CH-controlled AD.controlled X56 Incongruent) |        0.38 | 0.04 | [0.30, 0.47] |   8.72 | < .001
t_to_eta2(t=8.72,df_error=168)=.31



cont4 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical TS-controlled AD.controlled X71 Congruent) - (physical TS-controlled AD.controlled X71 Incongruent)'))
# (physical TS-controlled AD.controlled X71 Congruent) - (physical TS-controlled AD.controlled X71 Incongruent) |        0.41 | 0.05 | [0.32, 0.50] |   9.00 | < .001
t_to_eta2(t=9.00,df_error=168)=.33

cont4 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical TS-controlled AD.controlled X86 Congruent) - (physical TS-controlled AD.controlled X86 Incongruent)'))
# (physical TS-controlled AD.controlled X86 Congruent) - (physical TS-controlled AD.controlled X86 Incongruent) |        0.38 | 0.05 | [0.28, 0.48] |   7.33 | 0.005
t_to_eta2(t=7.33,df_error=168)=.24

cont5 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical AD-controlled AD.controlled X50 Congruent) - (physical AD-controlled AD.controlled X50 Incongruent)'))
# (physical TS-controlled AD.controlled X86 Congruent) - (physical TS-controlled AD.controlled X86 Incongruent) |        0.38 | 0.05 | [0.28, 0.48] |   7.33 | 0.005
t_to_eta2(t=7.33,df_error=168)=.24

cont6 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical AD-controlled AD.controlled X71 Congruent) - (physical AD-controlled AD.controlled X71 Incongruent)'))
# (physical AD-controlled AD.controlled X71 Congruent) - (physical AD-controlled AD.controlled X71 Incongruent) |        0.34 | 0.05 | [0.25, 0.43] |   7.28 | 0.006
t_to_eta2(t=7.28,df_error=168)=.24

cont7 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical AD-controlled TS.controlled X50 Congruent) - (physical AD-controlled TS.controlled X50 Incongruent)'))
# (physical AD-controlled TS.controlled X50 Congruent) - (physical AD-controlled TS.controlled X50 Incongruent) |        0.25 | 0.04 | [0.18, 0.32] |   6.96 | 0.024
t_to_eta2(t=6.96,df_error=168)=.22


cont8 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical AD-controlled AD.controlled X63 Incongruent) - (physical AD-controlled CH.controlled X63 Incongruent)'))
# (physical AD-controlled AD.controlled X63 Incongruent) - (physical AD-controlled CH.controlled X63 Incongruent) |       -0.21 | 0.03 | [-0.26, -0.16] |  -8.01 | < .001
t_to_eta2(t=8.01,df_error=168)=.28


cont9 <-subset(contrasts_five_way_interaction_table_not_split, (contrast == '(physical TS-controlled TS.controlled X50 Congruent) - (numerical-physical TS-controlled TS.controlled X50 Congruent)'))



#######################
# three way interaction
#######################
three_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Congruity)
contrasts_three_way_interaction_not_split <- pairs(three_way_interaction_not_split_into_simple_cong_effects)
contrasts_three_way_interaction_table_not_split <- parameters::model_parameters(contrasts_three_way_interaction_not_split)


cont15 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(physical TS-controlled Congruent) - (numerical-physical TS-controlled Congruent)'))
# (physical TS-controlled Congruent) - (numerical-physical TS-controlled Congruent) |        0.29 | 0.06 | [0.18, 0.40] |   5.09 | 0.007
t_to_eta2(t=5.09,df_error=168)=.13

cont16 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(physical TS-controlled Incongruent) - (numerical-physical TS-controlled Incongruent)'))

cont17 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(numerical-physical AD-controlled Congruent) - (numerical-physical TS-controlled Congruent)'))
cont19 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(physical CH-controlled Congruent) - (physical CH-controlled Incongruent)'))
# (physical CH-controlled Congruent) - (physical CH-controlled Incongruent) |        0.17 | 0.02 | [0.13, 0.22] |   7.17 | < .001
t_to_eta2(t=7.17,df_error=168)=.23




cont18 <-subset(contrasts_three_way_interaction_table_not_split, (contrast == '(colors TS-controlled Incongruent) - (numerical-physical TS-controlled Incongruent)'))
# (colors TS-controlled Incongruent) - (numerical-physical TS-controlled Incongruent) |        0.26 | 0.05 | [0.16, 0.35] |   5.28 | 0.003
t_to_eta2(t=5.28,df_error=168)=.14






#######################
# four way interaction
#######################
four_way_interaction_not_split_into_simple_cong_effects <- emmeans(fit, ~ Task * Train * Test * Congruity)
contrasts_four_way_interaction_not_split <- pairs(four_way_interaction_not_split_into_simple_cong_effects)
contrasts_four_way_interaction_table_not_split <- parameters::model_parameters(contrasts_four_way_interaction_not_split)

four_way_sigificant_effects <- subset(contrasts_four_way_interaction_table_not_split, (p<=0.05))
# filter sigificant_effects for contrast containing the string 'numerical-physical AD-controlled TS.controlled'
filter4 <- subset(four_way_sigificant_effects, grepl('numerical-physical TS-controlled TS.controlled', contrast))
sigificant_effects_filter_4 <- subset(filter4, grepl('numerical-physical TS-controlled AD.controlled', contrast))

cont13 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(numerical-physical AD-controlled AD.controlled Congruent) - (numerical-physical TS-controlled TS.controlled Congruent)'))
# (numerical-physical AD-controlled AD.controlled Congruent) - (numerical-physical TS-controlled TS.controlled Congruent) |        0.32 | 0.06 | [0.21, 0.44] |   5.60 | 0.035
t_to_eta2(t=5.60,df_error=168)=.16


cont15 <-subset(contrasts_four_way_interaction_table_not_split, (contrast == '(colors-numerical CH-controlled CH.controlled Congruent) - (colors-numerical TS-controlled CH.controlled Congruent)'))
# (colors-numerical AD-controlled CH.controlled Congruent) - (colors-numerical CH-controlled CH.controlled Congruent) |        0.25 | 0.04 | [0.18, 0.33] |   6.57 | < .001
t_to_eta2(t=6.57,df_error=170)=0.20


#######################
# four way interaction over all tests
#######################

four_way_interaction_effects <- emmeans(fit, ~ Task * Train * Ratio * Congruity)
contrasts_four_way_interaction_not_split1 <- pairs(four_way_interaction_effects)
contrasts_four_way_interaction_table_not_split1 <- parameters::model_parameters(contrasts_four_way_interaction_not_split1)


cont16 <-subset(contrasts_four_way_interaction_table_not_split1, (contrast == '(physical CH-controlled X50 Incongruent) - (numerical-physical CH-controlled X71 Incongruent)'))
