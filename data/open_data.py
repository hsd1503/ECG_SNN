import numpy as np
import matplotlib.pyplot as plt

loadData = np.load('./data/raw_data.npy',allow_pickle=1)
loadData2 = np.load('./data/raw_labels.npy',allow_pickle=1)

print("----type----")
print(type(loadData))
print("----shape----")
print(loadData.shape)
print("----data----")
print(loadData)

for x in range(5017,5056):
    plt.plot(loadData[x])
    print(x,loadData[x].shape,loadData2[x])
    plt.title(str(loadData2[x]))    
    plt.show()
