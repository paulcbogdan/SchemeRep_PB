#-------------

library(lme4)      # for fitting model
library(lmerTest)  # for getting df, t and p for fixed effects
library(optimx)    # needed for changing algorithm
require(dplyr) 
require(sjPlot)
require(robumeta)

#-------------

# fp = r'(C:\PycharmProjects\SchemeRep\HCP_gambling\rs_task_df.csv)'
# fp = r'(C:\PycharmProjects\SchemeRep\Study2B\rs_task_wl_df.csv)'
fp = r'(C:\PycharmProjects\SchemeRep\Study2B\rs_task_df_3432000.csv)'
# fp = r'(C:\PycharmProjects\SchemeRep\Study2B\rs_task_schaefer_df.csv)'
# fp = r'(C:\PycharmProjects\SchemeRep\Study2B\Study2B_rest_x_task_lmer_input.csv)'
fp = r'(C:\PycharmProjects\SchemeRep\Study2B\rs_task_schaefer_comb_df_36000.csv)'
# fp = r'(C:\PycharmProjects\SchemeRep\Study2B\rs_task_schaefer_df_R_500000.csv)'

df = read.csv(fp)



#-------------
df$task = scale(df$task)
df$rs = scale(df$rs)

mod <- lmer('rs ~ 1 + task + (1 | roi) + (1 + task | sn)', data=df,
            control = lmerControl(optimizer = "optimx", calc.derivs = FALSE, 
                                  optCtrl = list(method = "nlminb", 
                                                 starttests = FALSE, kkt = FALSE
                                  )),
            REML=F, verbose=100)
print(summary(mod))
# print(effectsize::standardize_parameters(mod))

#-------------
