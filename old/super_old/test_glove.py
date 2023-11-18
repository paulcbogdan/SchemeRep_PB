

fp = 'RSAmodels/example_deepNeuralNetworkScripts_from_Lifu/RSAmodel/glove.6B.200d.txt'
with open(fp, 'r') as f:
    lines = f.readlines()

lookfor = 'ferris wheel'

for ln in lines[:1000]:
    print(ln[:30])
    # print(ln)



