#-------------

library(lme4)      # for fitting model
library(lmerTest)  # for getting df, t and p for fixed effects
library(optimx)    # needed for changing algorithm
require(dplyr) 
require(sjPlot)
require(robumeta)

#-------------


fp = r'(C:\PycharmProjects\SchemeRep\Study2B\rs_task_wl_df_3432000.csv)'

df = read.csv(fp)


# #-------------
# df <- df %>%
#   group_by(sn) %>%
#   mutate(across(c(rs, task), ~ (. - mean(.)) / sd(.), .names = "z_score_{col}")) %>%
#   ungroup()


#-------------
# df$task = scale(df$task)
# df$rs = scale(df$rs)


# data <- data %>%
#   group_by(sn) %>%
#   mutate(across(c(rs, tas), ~ (. - mean(.)) / sd(.), .names = "z_score_{col}")) %>%
#   ungroup()


mod <- lmer('rs ~ 1 + task + (1 + task | sn) ', data=df,
            control = lmerControl(optimizer = "optimx", calc.derivs = FALSE,
                                  optCtrl = list(method = "nlminb",
                                                 starttests = FALSE, kkt = FALSE
                                  )),
            REML=F, verbose=100)
print(summary(mod))

# mod <- lmer('z_score_rs ~ 1 + z_score_task + (1 + z_score_task | sn)', data=df,
#             control = lmerControl(optimizer = "optimx", calc.derivs = FALSE,
#                                   optCtrl = list(method = "nlminb",
#                                                  starttests = FALSE, kkt = FALSE
#                                   )),
#             REML=F, verbose=100)
# print(summary(mod))
# print(effectsize::standardize_parameters(mod))

#-------------
