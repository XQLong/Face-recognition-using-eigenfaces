import cv2
from sklearn.model_selection import train_test_split

import eigenFace as ef
import numpy as np
import matplotlib.pyplot as plt

testef = ef.EigenFace(variance_pct=0.99,knn=1)
#训练集图像矩阵
data_train = testef.vector_matrix
#测试集图像矩阵
# data_test = testef.test_vector_matrix
#print(data_test.shape)

responses = []      #训练集样本标签
for name,img,label in testef.image_dictionary:
    responses.append(label)

#测试集样本标签
# test_tag = []
# for name,img,label in testef.test_imsge_dictionary:
#     test_tag.append(label)

def cal_pre(variance):
    testef.variance_pct = variance
    # 将数据集分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data_train.T, responses, test_size=0.1)
    testef.vector_matrix = x_train.T
    data_test = x_test.T;
    #对训练样本计算特征空间
    testef.get_eigen()
    testef.lamb = testef.eigen.lamb    #特征值
    testef.u = testef.eigen.u      #单位特征向量
    testef.variance_proportion = testef.eigen.variance_proportion      #归一化特征值
    testef.mean_vector = testef.eigen.mean_vector      #平均脸
        #训练集图像矩阵已被均值化处理

    #对测试数据进行均值处理
    for ii in range(data_test.shape[1]):
        data_test[:, ii] -= testef.mean_vector

    #根据重构阈值确定特征脸个数
    k = testef.get_number_of_components_to_preserve_variance(variance)

    #计算训练样在特征脸上的权重（映射到特征空间）
    weight_train = testef.getWeight4Training()

    #计算测试样本在特征脸上的权重
    weight_test = np.dot(testef.u.T, data_test)

    #knn进行距离分类识别
    knn = cv2.ml.KNearest_create()
    knn.train(weight_train[0:k,:].T.astype(np.float32),cv2.ml.ROW_SAMPLE,np.asarray(y_train,dtype=np.float32))
    # we have to discard the first predict result, since it has to be itself
    ret, results, neighbours2 ,dist = knn.findNearest(weight_test[0:k,:].T.astype(np.float32), 1)
    neighbours = neighbours2[:,0:]
    eval_data = []
    for idx, nb in enumerate(neighbours):
        neighbours_count = []
        for n in nb:
            neighbours_count.append(nb.tolist().count(n))
        vote = nb[neighbours_count.index(max(neighbours_count))]
        eval_data.append((vote, y_test[idx]))
        # print ("predict:%s, neight: %s, label: %d" % (str(vote),str(nb), responses[idx]))
    precision = testef.get_testeval(eval_data)
    return precision

varr_t = np.arange(0.50,1,0.01)
precisions = []
pre = 0.0
for i,var in enumerate(varr_t):
    for j in range(1,10):
        pre += cal_pre(var)
    precisions.append([])
    pre = pre/10
    precisions[i].append(pre)

plt.figure()
plt.grid(True)
plt.xlabel('variance')
plt.ylabel('Precision')

plt.plot(varr_t,precisions)
#precisions = np.asarray(precisions)
plt.show()