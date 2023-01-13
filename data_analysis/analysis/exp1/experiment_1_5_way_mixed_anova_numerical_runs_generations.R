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
master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/exp1/anova_df_for_R_31_12_22_generations.csv", header = TRUE, colClasses = c("factor", "factor", "factor", "numeric"))

summary(master)


# #5 way anova - dependent variable Genertions:
# (fit <- aov_ez('UNIQUE_SUBJECT_UID','Generations',master,
#                 between= c('Task', 'Train'),
#                 anova_table = list(es = 'pes'), ))
# #return="nice"
# summary(fit)
# failed because there is no variant between subjects

# group by (over individuals)
group_by_df = master %>% group_by(Task,Train) %>%
                   summarise(Generations = mean(Generations),
                             std = sd(Generations),
                             .groups = 'drop')
# adding an ID column
# group_by_df$Id <- seq.int(nrow(group_by_df))
# fit2 <- aov_ez(id='Id', dv='Generations', between=c('Task', 'Train'),data = group_by_df,
#   anova_table = list(es = 'pes'))
# summary(fit2)

group_to_keep <- group_by_df[,c("Task", "Train","Generations")]
head(group_to_keep,20)


############
# Visualize
###########
p_overall <- ggplot(group_to_keep, aes(x=Task, y=Generations, fill=Train)) +
  theme_classic2() + scale_fill_brewer(palette="Set3") +
  geom_bar(stat="identity",  colour = "black", position=position_dodge()) +
  ggtitle("Generations in Numerical and Physical Runs") +  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 20, family="Times New Roman", face="bold"))
p_overall



#this concats two columns into one
# group_to_keep <- group_by_df[,c("Task_Train","Generations")]



generations_count_vs_colors_count_df <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/genrations_exp1.csv", header = TRUE, colClasses = c( "numeric", "numeric", "numeric"))
generations_count_vs_colors_count_df.long <- generations_count_vs_colors_count_df %>%
  + gather(key = "group", value = "generations", count, colors_count)
head(generations_count_vs_colors_count_df.long, 10)
#   phys_property        group generations
# 1             1        count           9
# 2             2        count          10
# 3             3        count           3
# 4             1 colors_count           6
# 5             2 colors_count          19
# 6             3 colors_count           3
stat.test <- generations_count_vs_colors_count_df.long  %>%
  t_test(generations ~ group, paired = FALSE) %>%
  add_significance()

# A tibble: 1 × 9
#   .y.         group1       group2    n1    n2 statistic    df     p p.signif
#   <chr>       <chr>        <chr>  <int> <int>     <dbl> <dbl> <dbl> <chr>
# 1 generations colors_count count      3     3     0.555     2 0.635 ns
