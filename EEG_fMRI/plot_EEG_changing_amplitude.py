import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    t_h = np.pi * 9
    t = np.linspace(0, t_h, 1000)
    A = np.linspace(1, 0.2, 1000)
    # A = np.concatenate([A, [2] * 100, A[::-1]]) # how tf does copilot know
    x = np.sin(t + np.pi) * A
    # x2 = np.sin(t * 2) * (1 - A)
    y = -x


    # plot x axis
    plt.plot(t, np.zeros_like(t), 'k-', linewidth=0.5)
    plt.xlim(0, t_h)
    plt.plot(t, x, color='dodgerblue')
    # plt.plot(t, x + x2, color='blue')
    plt.plot(t, y, color='red')
    plt.gca().spines[['top', 'bottom', 'right', 'left']].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.yticks([-1, 0, 1], ['100%\na', '0', '100%\nb'])
    plt.savefig('result_pics/other/fluc_drawing.png', dpi=300)
    plt.show()





