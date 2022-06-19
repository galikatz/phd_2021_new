# Title     : Anova - from size to counting
# Created by: gali.k
# Created on: 15/01/2022

# install.packages(c("ggplot2", "ggpubr", "tidyverse", "broom", "AICcmodavg"))
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(broom)
#library(AICcmodavg)

size_data <- read.csv("data_analysis/rawdata/aggregated/agg_results_size_4_1_22.csv",
                      header = TRUE, colClasses = c("factor", "factor", "factor", "factor", "factor","factor",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric",
                                                    "factor", "factor", "factor", "factor"))
count_data <- read.csv("data_analysis/rawdata/aggregated/agg_results_count_4_1_22.csv",
                      header = TRUE, colClasses = c("factor", "factor", "factor", "factor", "factor","factor",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                    "numeric","numeric","numeric","numeric","numeric",
                                                    "factor", "factor", "factor", "factor"))
size_count_data <- read.csv("data_analysis/rawdata/aggregated/agg_results_size_count_4_1_22.csv",
                       header = TRUE, colClasses = c("factor", "factor", "factor", "factor", "factor","factor",
                                                     "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                     "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                     "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                     "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                     "numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric",
                                                     "numeric","numeric","numeric","numeric","numeric",
                                                     "factor", "factor", "factor", "factor"))
total <- rbind(size_data, count_data, size_count_data)
#summary(total)

simple_df <- total$Equate
simple_df <- total$Task
#simple_df <- total$"Ratio 50 Congruent Training Accuracy"
head(simple_df,10)







