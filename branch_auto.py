from pathlib import Path
from time import time

from Utils.pickle_wrap_funcs import pickle_wrap
from marinate.pkld import pkld
import matplotlib.pyplot as plt


# from pkld import pkld


def make_auto_branch(func_dir):
    t_st = time()
    fps = list(Path(func_dir).glob('*.pkl'))
    print(f'time needed to get all pkl files: {time() - t_st:.2f} s')
    print(len(fps))

# @pkld(branch_factor=100)
def pkl_test(a=0):
    return a


def run_pkl_test():
    t_st = time()
    ts = []

    for i in range(10_000):
        if i % 1000 == 0 and i > 0:
            t = time() - t_st
            ts.append(t)
            print(f'1000 calls: {i=}, {t=:.4f} s')
            plt.rcParams.update({'font.size': 12})
            plt.plot(ts, color='r')
            plt.xlabel('Time needed for 1000 calls')
            plt.ylabel('Seconds (s)')
            plt.title('pickle_wrap read')
            plt.gca().spines[['right', 'top']].set_visible(False)
            plt.show()
            t_st = time()

        pkl_test(i)

        # pickle_wrap(pkl_test, kwargs={'a': i},
        #             verbose=-1, easy_override=False)



if __name__ == '__main__':
    run_pkl_test()
    # make_auto_branch(r'C:\PycharmProjects\SchemeRep\cache\get_sn_fp_llama_RSM_')