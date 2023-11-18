import scipy.io
from scipy import io

# Encoding task
# Counterbalanced into one of three groups based on subject_number % 3
# Each group encounters the same set of stimuli. However, the stimuli are
#   randomized between participants.
# Stimuli are divided into three runs of 38 trials each

fp = 'SchemRep_tasks/PTBtasks/results/S138_run1.mat'

# Resultts (.mat) files are organized where
# mat['pdata'][0][0][7] is the object name
# mat['pdata'][0][0][8] is the scene name
# mat['pdata'][0][0][9] is 1 for incongruent, 2 for neutral, and 3 for congruent

mat = scipy.io.loadmat(fp)
print(mat)