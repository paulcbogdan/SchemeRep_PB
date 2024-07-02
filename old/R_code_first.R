library(glue)
library(lme4)
library(lmerTest)
library(Matrix)

setwd(r'(H:\PycharmProjects_H\SchemeRep)')

# ----
sess = 'obj7_fMRI'
fp_in = glue(r'(csv_out/{sess}_super_first.csv)')
df = read.csv(fp_in)

formula = 'vv_shortest ~ 1 + dd + DP_hemi + DA_hemi +'
' vv + dv_ant + dv_pos + inc'
'+ pd_M + ad_M + pv_M + av_M'
'+ (1  + dd + DP_hemi + DA_hemi | sn)'

mod_n <- lmer(dd_clustering ~ 1 + dd + DP_hemi + DA_hemi + 
                vv + dv_ant + dv_pos + inc + 
                pd_M + ad_M + pv_M + av_M + 
                (1  + dd + DP_hemi + DA_hemi | sn), 
              data=df,
              control = lmerControl(optimizer = "optimx", 
                                    calc.derivs = FALSE,
                                    optCtrl = list(method = "L-BFGS-B", 
                                                   starttests = FALSE, 
                                                   kkt = FALSE)),
              REML=F)

print(summary(mod_n))

