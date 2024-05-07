# df_sn = get_trial_info('102')
# semantic = True
# d_vecs = prep_vecs(True, semantic)
# RSM_vis = get_stim_RDM(df_sn, d_vecs, obj_only=True,
#                        dist='corr')
# tril = np.tril_indices(RSM_vis.shape[0], k=-1)
# RSM_vis_flat = RSM_vis[tril]
# semantic = False
# d_vecs = prep_vecs(True, semantic)
# RSM_sem = get_stim_RDM(df_sn, d_vecs, obj_only=True,
#                        dist='corr')
# RSM_sem_flat = RSM_sem[tril]
# r, p = stats.spearmanr(RSM_vis_flat, RSM_sem_flat)
# plt.imshow(RSM_vis)
# plt.show()
# plt.imshow(RSM_sem)
# plt.show()
# print(f'{r=:.3f}')
# quit()
#
# semantic = True
# d_vecs = prep_vecs(True, semantic)
# vecs_all = np.array([d_vecs[obj] for obj in df_sn['obj']]).T
# corr = np.corrcoef(vecs_all)
# corr[np.diag_indices_from(corr)] = np.nan
# plt.imshow(corr)
# plt.colorbar()
# plt.show()
#
# pca = decomposition.PCA()
# pca.fit(vecs_all)
# vecs_PCA = pca.transform(vecs_all)
#
# # print(vecs_PCA.shape)
# # quit()
#
# exp_var_ratios = pca.explained_variance_ratio_
#
# # Calculate the cumulative sum of the explained variance ratios
# cum_sum_explvar = np.cumsum(exp_var_ratios)
#
# # Create a vector for the number of components
# n_components = np.arange(1, len(exp_var_ratios) + 1)
#
#
# corr = np.corrcoef(vecs_PCA[:, :10])
# corr[np.diag_indices_from(corr)] = np.nan
# plt.imshow(corr)
# plt.colorbar()
# plt.show()
# quit()
#
# # Plot the scree plot
# plt.figure(figsize=(8, 6))
# plt.plot(n_components, exp_var_ratios * 114, marker='o')
# plt.plot(n_components, cum_sum_explvar, marker='o')
# # plt.xticks(n_components)
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Scree Plot')
# plt.legend(['Individual', 'Cumulative'])
# plt.show()
#
# # quit()

# vecs_all = stats.zscore(vecs_all, axis=1)