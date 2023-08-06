import numpy as np
import matplotlib.pyplot as plt
classes = ['0','1','2','3']
confusion_matrix = np.array([(476,4,23,6),(6,59,20,1),(46,10,172,1),(9,3,7,12)],dtype=np.float64)

# 476	4	23	6
# 6	    57	20	1
# 46	10	172	1
# 9	    3	7	12

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(4)] for i in range(4)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]))   #显示对应的数字

# plt.ylabel('Real label')
# plt.xlabel('Prediction')
plt.tight_layout()
plt.show()
