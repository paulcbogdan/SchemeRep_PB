from Utils.atlas_funcs import get_atlas
from connRSA.old.conn_ISPC import run_settings_ISPC
from connRSA.conn_regress import get_title
from connRSA.old_Oct29.conn_report import report_results
from connRSA.single_trial_conn import run_settings
from old.plot_gen import my_plot_surf
from Utils.pickle_wrap_funcs import pickle_wrap


# os.chdir(r'/')

# import warnings
# warnings.filterwarnings("error")

def do():
    RSA = True
    ISPC = False
    ERS_alt = False

    semantic = False
    trial_similarity = 'euc'# 'seuclidean' #
    second_order = 'spear'
    RDM_method = 'by_run'#'clever_std_complex_mean'
    four_tasks = '7'
    stdize_by_run = True if trial_similarity == 'euc' else False
    stdize_by_run = False

    conn = 'BOLD_avg_norm_half'
    # assert not 'avg' in conn and 'norm' in conn
    # conn = 'prod'

    dir_results = r'cache/conn_RSA'
    if ISPC:
        kwargs = {'trial_similarity': trial_similarity,
                  'four_tasks': four_tasks,
                  'do_networks': False,
                  'conn': conn,
                  }
        results_bold = pickle_wrap(run_settings_ISPC, None,
                                   kwargs=kwargs,
                                   easy_override=False,
                                   cache_dir=dir_results)
    else:
        kwargs = {'semantic': semantic,
                  'trial_similarity': trial_similarity,
                  'second_order': second_order, 'RDM_method': RDM_method,
                  'stdize_by_run': stdize_by_run,
                  'four_tasks': four_tasks,
                  'do_networks': False,
                  'conn': conn,
                  'RSA': RSA,
                  'age': 'healthy'
                  }
        results_bold = pickle_wrap(run_settings, None,
                                   kwargs=kwargs,
                                   easy_override=True, verbose=1,
                                   cache_dir=dir_results)
    ts, _ = report_results(results_bold, do_lmer=False, ISPC=ISPC)

    title, _, _ = get_title(RSA, ISPC,
                            kwargs['semantic'] if 'semantic' in kwargs else None,
                            conn,'regress_row_str', None)
    print(f'{title=}')
    atlas = get_atlas()
    my_plot_surf(ts, atlas, title, fp_out=None,
                 neg='', pos='', thresh=1.65, vmax=4)



if __name__ == '__main__':
    do()










