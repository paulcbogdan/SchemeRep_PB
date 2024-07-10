
import numpy as np
from matplotlib import pyplot as plt
from nilearn.glm.first_level import compute_regressor
from scipy import ndimage


def get_hrf_():
    onset, amplitude, duration = 0.0, 1.0, 0.1
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
    # time_length = 21
    frame_times = np.arange(40)# * 2.1
    # print(frame_times)
    # quit()
    signal, _labels = compute_regressor(
        exp_condition,
        'spm',
        frame_times,
        con_id="main",
        oversampling=50,
        # min
    )
    # print(signal.shape)
    # print(signal)
    # quit()
    return signal[:, 0]

HRF = get_hrf_()
print(HRF)

l = np.zeros(61)
l[22] = 10
l[20:22] = -1
l[23:30] = -1

l_c = ndimage.convolve1d(l, HRF,
                         origin=-HRF.shape[0] // 2, axis=0)
# print(l_c)

fig, ax = plt.subplots(2, 1)
# ax[0].plot(l)
l_plotting = []

x = []
y = []
for i, val in enumerate(l):
    if val != 0:
        if val != l[i-1]:
            x.append(i)
            y.append(0)
        x.append(i)
        y.append(val)
        x.append(i+1)
        y.append(val)
    else:
        x.append(i)
        y.append(val)

ax[0].set_title('Regressor')
ax[0].plot(x, y, color='orange',)
ax[0].set_xlim(0, 60)

ax[1].set_title('HRF convolved')
ax[1].plot(l_c, color='dodgerblue')
ax[1].set_xlabel('Don\'t worry about the time scale')
ax[1].set_xlim(0, 60)
plt.tight_layout()
plt.show()

# print(hrf)
# quit()
