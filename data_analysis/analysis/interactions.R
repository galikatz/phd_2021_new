master <- read.csv("/Users/gali.k/phd/phd_2021/data_analysis/analysis/anova_df_for_R_new.csv", header = TRUE,
                      colClasses = c("numeric", "factor", "factor", "factor", "factor", "factor", "factor", "numeric"))


noise.lm <- lm(noise/10 ~ size * type * side, data = master)
anova(noise.lm)