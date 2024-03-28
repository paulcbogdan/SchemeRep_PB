from statsmodels.stats.multitest import multipletests
import numpy as np

ps = [.0005, .01, .05, .0005, .01, .03, .007, .005, .01, .05, .05, .008, .007,
      .04, .007, .005, .005, .008, .0005]
ps += [.5] * 190
ps = np.array(ps)
ps /= 3

reject, corr, _, _ = multipletests(ps, method='fdr_bh')
print(f'{reject.sum()} rejections')
print(f'{corr=}')




