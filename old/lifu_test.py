from scipy import io

from stim import get_stim_RDM_lifu

# good
fp0 = r'C:\PycharmProjects_C\SchemeRep\old\RSAmodels\example_deepNeuralNetworkScripts_from_Lifu\RSAmodel\modelRDMs' \
      r'\RSM_VGG16_PCA.mat'

# bad
fp1 = r'C:\PycharmProjects_C\SchemeRep\old\RSAmodels\example_deepNeuralNetworkScripts_from_Lifu\RSAmodel\cornet_models' \
      r'\RSM_VGG16_PCA.mat'

fp_in = r'C:\PycharmProjects_C\SchemeRep\old\RSAmodels' \
        r'\example_deepNeuralNetworkScripts_from_Lifu\RSAmodel\modelRDMs' \
        r'\RSM_VGG16_PCA.mat'

fp_sem = r'C:\PycharmProjects_C\SchemeRep\old\RSAmodels\W2Vsemantic_RDM.mat'
mat = io.loadmat(fp_sem)
RDM_stim = mat['R']

print(RDM_stim)
quit()

mat0 = io.loadmat(fp0)
print(mat0['R'])
quit()
mat1 = io.loadmat(fp1)
print(mat1)