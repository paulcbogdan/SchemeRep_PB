from collections import defaultdict
from datetime import datetime
from random import random

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from atlas_utils import get_atlas
from conn_utils import get_ROI_vecs_wrap, get_conn_vecs, get_trial_x_trial
from organize_bhv import get_trial_info
from org_sns import get_sns
from tqdm import tqdm

from single_trial_conn import prep_fps, run_settings_healthy
from conn_report import report_results
from old.networks import prep_networks
from utils import pickle_wrap
import warnings
from colorama import Fore
import os

os.chdir(r'E:\PycharmProjects_E\SchemeRep')


def ISPC(atlas, sns, fp='bl2_fMRI', conn='euc', combine_regions=False, split=False,
		 networks=False,  trial_similarity='corr', ):
	BOLD = conn == 'BOLD'
	cross_region = 'cross_' in conn
	if 'cross_' in conn:
		conn = conn.replace('cross_', '')


	ROI2vecs_all_sn = defaultdict(list)

	org_by_region = (not BOLD) or (networks)
	# print(f'{org_by_region=}')
	# quit()

	for sn in sns:
		df_sn = get_trial_info(sn, verbose=-1)
		ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
									 networks=networks,
									 org_by_region=org_by_region,
									 cross_region=cross_region, conn=conn,
									 combine_regions=combine_regions,
									 )

		for ROI, vecs in ROI2vecs.items():
			vecs_BOLD = ROI2vecs[ROI]
			# print(f'{ROI} | {vecs_BOLD.shape=}')
			if BOLD or cross_region:
				vecs = vecs_BOLD
			else:
				vecs = get_conn_vecs(vecs_BOLD, conn=conn)
			vecs = vecs[df_sn['obj'].argsort(), :]
			ROI2vecs_all_sn[ROI].append(vecs)

	score_by_stim = []
	sizes = []
	for ROI, ROI_vecs in ROI2vecs_all_sn.items():
		ROI_vecs = np.array(ROI_vecs)
		print('test')
		# quit()

		sn_Ms = np.nanmean(ROI_vecs, axis=1)[:, None, :]
		sn_SDs = np.nanstd(ROI_vecs, axis=1)[:, None, :]
		ROI_vecs = (ROI_vecs - sn_Ms) / sn_SDs

		size = np.sum(~np.isnan(ROI_vecs)) / (ROI_vecs.shape[0] *
											  ROI_vecs.shape[1])
		# bad_sns_base = np.array([False, False, False, False, False, False,
		# 						 False, False, False, False, False, False,
		# 						 False, False, False, False, False, False,
		# 						 False, False, False, False, False, False,
		# 						 False, False, False, False, False, False,
		# 						 False,  True, False, False, False, False,
		# 						 False, False, False, False, False, False,
		# 						 False, False, False, False, False, False,
		# 						 False, False, False, False, False, False,
		# 						 False, False, False])

		if org_by_region:
			bad_sns = np.isnan(ROI_vecs).any(axis=(1, 2))# > 0.5
			print(f'{ROI} | Number of bad sns: {np.sum(bad_sns)}')
			ROI_vecs = ROI_vecs[~bad_sns]
			keeps = ~np.isnan(ROI_vecs).any(axis=(0, 1))
			nan_prop = np.sum(np.isnan(ROI_vecs)) / np.prod(ROI_vecs.shape)
			n_good = np.sum(keeps)
			if n_good < 10:
				warn_str = f'Barely any good voxels for {ROI}. {n_good=}, {nan_prop=:.3f}'
				warnings.warn(warn_str)
				score_by_stim.append(np.full(ROI_vecs.shape[1], np.nan))
				sizes.append(0)
				continue
		else:
			print(ROI_vecs.shape)
			bad_voxels = np.isnan(ROI_vecs).mean(axis=(0, 1)) > 0.5
			ROI_vecs = ROI_vecs[..., ~bad_voxels]
			bad_sns = np.isnan(ROI_vecs).mean(axis=(1, 2)) > 0.25
			ROI_vecs = ROI_vecs[~bad_sns]

			M = np.nanmean(ROI_vecs, axis=2)
			for i in range(ROI_vecs.shape[0]):
				for j in range(ROI_vecs.shape[1]):
					nans = np.isnan(ROI_vecs[i, j, :])
					ROI_vecs[i, j, nans] = M[i, j]
			keeps = ~np.isnan(ROI_vecs).any(axis=(0, 1))

			# plt.imshow(np.isnan(ROI_vecs).mean(axis=-1), aspect='auto')
			# plt.colorbar()
			# plt.show()
			# print(bad_sns)
			# print(np.mean(bad_voxels))
			# quit()

		sizes.append(size)
		ROI_same_scores = []
		ROI_else_scores = []
		for stim_j in tqdm(range(ROI_vecs.shape[1]), desc='ISPC'):
			stim_data = ROI_vecs[:, stim_j, :]
			stim_data = stim_data[:, keeps]

			ISPC_matrix = get_trial_x_trial(stim_data,
											trial_similarity=trial_similarity)
			ISPC_triangle = ISPC_matrix[np.tril_indices_from(ISPC_matrix, k=-1)]
			M_similarity = np.mean(ISPC_triangle)
			ROI_same_scores.append(M_similarity)

			stim_else_scores = []
			for stim_j2 in list(range(ROI_vecs.shape[1])):
				if stim_j2 == stim_j:
					continue
				# if random() > 0.1:
				# 	continue
				stim_data2 = ROI_vecs[:, stim_j2, :]
				stim_data2 = stim_data2[:, keeps]

				ISPC_matrix = get_trial_x_trial(stim_data, stim_data2,
											   trial_similarity=trial_similarity)
				ISPC_matrix[np.eye(ISPC_matrix.shape[0], dtype=bool)] = np.nan
				M_similarity2 = np.nanmean(ISPC_matrix)
				stim_else_scores.append(M_similarity2)

			M_similarity_else = np.mean(stim_else_scores)
			ROI_else_scores.append(M_similarity_else)

			# TODO: ISPC_IRAF = np.mean(ISPC_triangle, axis=1)


		ROI_same_scores = np.array(ROI_same_scores)
		ROI_else_scores = np.array(ROI_else_scores)
		ROI_scores = ROI_same_scores - ROI_else_scores


		ROI_M = np.mean(ROI_scores)
		ROI_SD = np.std(ROI_scores)
		ROI_SE = ROI_SD / np.sqrt(len(ROI_same_scores))
		ROI_t = ROI_M / ROI_SE
		ROI_p = stats.t.sf(np.abs(ROI_t), len(ROI_same_scores) - 1)# * 2
		print(f'{ROI} ({ROI_M:.5f}), t={ROI_t:.2f}, p={ROI_p:.3f}')
		score_by_stim.append(ROI_scores)
		# quit()
		# scores.append(ROI_M)
	score_by_stim = np.array(score_by_stim)
	return score_by_stim, sizes,

def run_settings_ISPC(four_tasks=True, conn='euc', combine_regions=False,
					  split=False, do_networks=1, trial_similarity='corr',
					  age=1):
	settings = locals().copy()
	print(f'Run settings ISPC: {settings=}')
	atlas = get_atlas(combine_regions=combine_regions,
					  combine_bilateral=False,
					  split=split, split_code='xyz')

	if do_networks:
		do_networks = prep_networks(do_networks)
		keys = list(do_networks)
	else:
		do_networks = None
		keys = atlas['tick_labels']
	age2sn = get_sns('all', sh=False)
	sns = age2sn[age]
	sns = [sn for sn in sns if int(sn) not in [230, 234, 239]]  # TODO: ask SH to re-run

	results = {'networks': do_networks, 'keys': keys, 'sns': sns}

	fps = prep_fps(four_tasks)
	scores_by_fp_ROI_stim, size_by_ROI = [], []
	for fp in fps:
		print(f'--------- {fp} ---------')
		# if 'obj' not in fp: continue
		scores, sizes = ISPC(atlas, sns, fp=fp, conn=conn,
							 combine_regions=combine_regions,
							 split=split, networks=do_networks,
							 trial_similarity=trial_similarity)
		scores_by_fp_ROI_stim.append(scores)

		size_by_ROI.append(sizes)

	scores_by_fp_ROI_stim = np.array(scores_by_fp_ROI_stim)
	scores_by_stim_fp_ROI = np.transpose(scores_by_fp_ROI_stim, (2, 1, 0))
	scores_by_ROI_stim = np.mean(scores_by_fp_ROI_stim, axis=0)
	size_by_ROI = np.array(size_by_ROI)

	results['settings'] = settings
	results['scores'] = scores_by_ROI_stim.T
	results['sizes'] = size_by_ROI
	results['scores_by_ROI'] = scores_by_stim_fp_ROI
	return results
#
# def run_settings_ISPC_healthy(settings):
# 	# results_both = run_settings_healthy(settings, run_settings_ISPC)
# 	dir_results = r'cache/conn_RSA'
#
# 	settings1 = settings.copy()
# 	del settings1['age']
# 	results1 = pickle_wrap(run_settings_ISPC, None, kwargs=settings1,
# 						   easy_override=True, verbose=0,
# 						   cache_dir=dir_results)
# 	print(f'{Fore.CYAN}Young people:{Fore.RESET}')
# 	report_results(results1, ISPC=True)
#
# 	settings2 = settings.copy()
# 	settings2['age'] = 2
# 	results2 = pickle_wrap(run_settings_ISPC, None, kwargs=settings2,
# 						   easy_override=True, verbose=0,
# 						   cache_dir=dir_results)
# 	print(f'{Fore.LIGHTYELLOW_EX}Old people:{Fore.RESET}')
# 	report_results(results2, ISPC=True)
#
# 	results_both = {'networks': results1['networks'],
# 					'keys': results1['keys'],
# 					'sns': results1['sns'] + results2['sns'],
# 					'scores': np.concatenate([results1['scores'],
# 											  results2['scores']]),
# 					'sizes': np.concatenate([results1['sizes'],
# 											 results2['sizes']]),
# 					'scores_by_ROI': np.concatenate([results1['scores_by_ROI'],
# 													 results2['scores_by_ROI']]),
# 					'settings': settings}
# 	return results_both

def run_ISPC(four_tasks=True, conn='euc', combine_regions=False,
			 split=False, do_networks=0, trial_similarity='corr',
			 age=1):
	settings = locals().copy()
	# OA = age == 2
	# if OA:
	# 	assert isinstance(four_tasks, str) and '3_' in four_tasks, 'Bad OA'
	# else:
	# 	del settings['age']
	assert not (combine_regions and split), 'cannot combine and split'
	assert (not combine_regions) or do_networks
	# assert not (conn == 'BOLD' and do_networks), 'Not conn=BOLD and do networks'
	if do_networks == 3:
		if 'cross' not in conn:
			settings['conn'] = f'cross_{conn}'
			print(f'Missing \"cross_\" for networks 3, Changed conn to {conn}')
	print(f'Before: {settings=}')
	dir_results = r'cache/conn_RSA'

	results = pickle_wrap(run_settings_ISPC, None, kwargs=settings,
						  easy_override=False, cache_dir=dir_results)
	print(f'{Fore.RED}Finished!{Fore.RESET}')
	report_results(results, ISPC=True)

def run_ISPC_toggle():
	trial_similarity_toggle = ['corr']
	four_tasks_toggle = ['7', '8']#, '3_3']
	conn_toggle = ['prod']
	# conn_toggle = ['BOLD']
	split_toggle = [False]
	age = 'healthy'

	for conn in conn_toggle:
		for four_tasks in four_tasks_toggle:
			# for do_networks in [1,  3, 4, 5, 6, 8, 9, False, 2, 7]:
			for do_networks in [16]:
				for split in split_toggle:
					for trial_similarity in trial_similarity_toggle:
						try:
							run_ISPC(conn=conn,
									 trial_similarity=trial_similarity,
									 four_tasks=four_tasks, split=split,
									 do_networks=do_networks,
									 age=age)
						except AssertionError as e:
							print(f'Assertion no bueno: {e}')
							pass

if __name__ == '__main__':
	run_ISPC_toggle()




