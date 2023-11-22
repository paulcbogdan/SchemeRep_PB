from collections import defaultdict
from random import random

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from atlas_utils import get_atlas
from conn_utils import get_ROI_vecs_wrap, get_conn_vecs, get_trial_x_trial
from organize_bhv import get_all_sns, get_trial_info
from tqdm import tqdm

from single_trial_conn import prep_fps, prep_networks, report_results
from utils import pickle_wrap
import warnings

def ISPC(atlas, sns, fp='bl2_fMRI', conn='euc', combine_regions=False, split=False,
		 networks=False,  trial_similarity='corr', ):
	BOLD = conn == 'BOLD'
	cross_region = 'cross_' in conn
	if 'cross_' in conn:
		conn = conn.replace('cross_', '')


	ROI2vecs_all_sn = defaultdict(list)

	for sn in sns:
		df_sn = get_trial_info(sn)
		ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
									 networks=networks,
									 org_by_region=not BOLD,
									 cross_region=cross_region, conn=conn,
									 combine_regions=combine_regions,
									 )
		for ROI, vecs in ROI2vecs.items():
			vecs_BOLD = ROI2vecs[ROI]
			if BOLD or cross_region:
				vecs = vecs_BOLD
			else:
				vecs = get_conn_vecs(vecs_BOLD, conn=conn)
			vecs = vecs[df_sn['obj'].argsort(), :]

			ROI2vecs_all_sn[ROI].append(vecs)

	score_by_stim = []
	sizes = []
	for ROI, ROI_vecs in ROI2vecs_all_sn.items():
		# if 'Occ' not in ROI and 'LOC' not in ROI: continue
		# if 'LOC' not in ROI: continue
		# if ROI != 'LOC' and ROI != 'EVC': continue
		ROI_vecs = np.array(ROI_vecs)

		sn_Ms = np.nanmean(ROI_vecs, axis=1)[:, None, :]
		sn_SDs = np.nanstd(ROI_vecs, axis=1)[:, None, :]
		ROI_vecs = (ROI_vecs - sn_Ms) / sn_SDs

		size = np.sum(~np.isnan(ROI_vecs)) / (ROI_vecs.shape[0] *
											  ROI_vecs.shape[1])
		keeps = ~np.isnan(ROI_vecs).any(axis=(0, 1))
		nan_prop = np.sum(np.isnan(ROI_vecs)) / np.prod(ROI_vecs.shape)
		n_good =  np.sum(keeps)
		if n_good < 10:
			warn_str = f'Barely any good voxels for {ROI}. {n_good=}, {nan_prop=:.3f}'
			warnings.warn(warn_str)
			score_by_stim.append(np.full(ROI_vecs.shape[1], np.nan))
			sizes.append(0)
			continue
		sizes.append(size)
		# print(f'{ROI}, {vecs_all.shape=}')
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
				# stim_data2 = np.repeat(np.array(list(range(30)))[:, None],
				# 					  stim_data.shape[1], axis=1)
				ISPC_matrix = get_trial_x_trial(stim_data, stim_data2,
											   trial_similarity=trial_similarity)
				ISPC_matrix[np.eye(ISPC_matrix.shape[0], dtype=bool)] = np.nan
				M_similarity2 = np.nanmean(ISPC_matrix)
				stim_else_scores.append(M_similarity2)
				# plt.imshow(ISPC_matrix)
				# plt.title(f'{M_similarity2=:.5f}')
				# plt.colorbar()
				# plt.show()
				# quit()
				# ISPC_triangle2 = ISPC_matrix[np.triu_indices_from(ISPC_matrix, k=-1)]

			M_similarity_else = np.mean(stim_else_scores)
			ROI_else_scores.append(M_similarity_else)
		ROI_same_scores = np.array(ROI_same_scores)
		# print(f'{ROI_same_scores=}')
		ROI_else_scores = np.array(ROI_else_scores)
		# print(f'{ROI_else_scores=}')
		ROI_scores = ROI_same_scores - ROI_else_scores
		# print(f'{ROI_scores=}')


		ROI_M = np.mean(ROI_scores)
		ROI_SD = np.std(ROI_scores)
		ROI_SE = ROI_SD / np.sqrt(len(ROI_same_scores))
		ROI_t = ROI_M / ROI_SE
		ROI_p = stats.t.sf(np.abs(ROI_t), len(ROI_same_scores) - 1) * 2
		print(f'{ROI} ({ROI_M:.5f}), t={ROI_t:.2f}, p={ROI_p:.3f}')
		score_by_stim.append(ROI_scores)
		# quit()
		# scores.append(ROI_M)
	score_by_stim = np.array(score_by_stim)
	return score_by_stim, sizes,

def run_settings_ISPC(four_tasks=True, conn='euc', combine_regions=False,
					  split=False, do_networks=1, trial_similarity='corr'):
	settings = locals().copy()
	atlas = get_atlas(combine_regions=combine_regions,
					  combine_bilateral=False,
					  split=split, split_code='xyz')

	if do_networks:
		do_networks = prep_networks(do_networks)
		keys = list(do_networks)
	else:
		do_networks = None
		keys = atlas['tick_labels']
	age2sn = get_all_sns(ret=True)
	sns = age2sn[1]

	results = {'networks': do_networks, 'keys': keys, 'sns': sns}

	fps = prep_fps(four_tasks)
	scores_by_fp_ROI_stim, size_by_ROI = [], []
	for fp in fps:
		print(f'--------- {fp} ---------')
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

def run_ISPC(four_tasks=True, conn='euc', combine_regions=False,
			 split=False, do_networks=0, trial_similarity='corr'):
	settings = locals().copy()
	assert not (combine_regions and split), 'cannot combine and split'
	assert (not combine_regions) or do_networks
	assert not (conn == 'BOLD' and do_networks), 'Not conn=BOLD and do networks'
	if do_networks == 3:
		if 'cross' not in conn:
			settings['conn'] = f'cross_{conn}'
			print(f'Missing \"cross_\" for networks 3, Changed conn to {conn}')
	print(f'Before: {settings=}')
	dir_results = r'cache/conn_RSA'
	results = pickle_wrap(None, run_settings_ISPC, kwargs=settings,
						  cache_dir=dir_results, easy_override=False)
	print(f'Finished!')
	report_results(results)

def run_ISPC_toggle():
	trial_similarity_toggle = ['corr']
	four_tasks_toggle = [True, False]
	conn_toggle = ['euc']
	# conn_toggle = ['BOLD']
	split_toggle = [False]

	for conn in conn_toggle:
		for four_tasks in four_tasks_toggle:
			for do_networks in [8]:
				for split in split_toggle:
					for trial_similarity in trial_similarity_toggle:
						try:
							run_ISPC(conn=conn,
										 trial_similarity=trial_similarity,
										 four_tasks=four_tasks, split=split,
										 do_networks=do_networks,)
						except AssertionError as e:
							print(f'Assertion no bueno: {e}')
							pass

if __name__ == '__main__':
	run_ISPC_toggle()




