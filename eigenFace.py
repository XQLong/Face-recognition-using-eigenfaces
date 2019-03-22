from __future__ import division
import os
import numpy as np
import scipy.stats as ss
import cv2
import matplotlib.pyplot as plt
import traitlets.utils.bunch as bunch
import fnmatch
import re
import collections

# All images are supposed to be the same size, say, N*L
IMAGE_DIR = "E:\\stu\\Machine_Learning\\Python_Ecosystem\\workplace\\Face_recog_eigenfaces\\training_data"
TEST_DIR  = "E:\\stu\\Machine_Learning\\Python_Ecosystem\\workplace\\Face_recog_eigenfaces\\test_data"


class EigenFace(object):
    # load images, and start other processing
    def __init__(self, image_path=IMAGE_DIR,test_image_path=TEST_DIR,suffix="*.*",variance_pct=0.99,knn=5):
        # the least variance percentage we want the top K eigen vector to cover.
        self.variance_pct = variance_pct
        # don't use the top k eigen vectors
        self.knn = knn
        # the original images corresponding to its name
        self.image_dictionary = []
        self.test_imsge_dictionary = []
        #读取训练数据集图片
        image_names = []
        for root, dirnames, filenames in os.walk(image_path):
            for filename in fnmatch.filter(filenames, suffix):
                image_names.append(os.path.join(root, filename))
        # image_names = [image for image in os.listdir(image_path) if not image.startswith('.')]
        for idx,image_name in enumerate(image_names):
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            if idx == 0:
                # 计算图像矩阵的大小M（假设为M = w*h）
                self.imgShape = img.shape
                print(self.imgShape)
                # 初始化一个M行，S(所有样本图片的个数)列的矩阵
                self.vector_matrix = np.zeros((self.imgShape[0]*self.imgShape[1], len(image_names)),dtype=np.float64)
            #img = cv2.pyrDown(img)
            self.image_dictionary.append((image_name,img,self.getClassFromName(image_name)))
            #将图像矩阵展开为一列
            self.vector_matrix[:,idx] = img.flatten()

        # 读取测试数据集图片
        test_image_names = []
        for root, dirnames, filenames in os.walk(test_image_path):
            for filename in fnmatch.filter(filenames, suffix):
                test_image_names.append(os.path.join(root, filename))
        # image_names = [image for image in os.listdir(image_path) if not image.startswith('.')]
        for idx, image_names in enumerate(test_image_names):
            img = cv2.imread(image_names, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            if idx == 0:
                # 初始化一个M行，S(所有样本图片的个数)列的矩阵
                self.test_vector_matrix = np.zeros((self.imgShape[0] * self.imgShape[1], len(test_image_names)), dtype=np.float64)
            # img = cv2.pyrDown(img)
            self.test_imsge_dictionary.append((image_names, img, self.getClassFromName(image_names)))
            # 将图像矩阵展开为一列
            self.test_vector_matrix[:, idx] = img.flatten()

        #print(self.test_imsge_dictionary)


        subjects = set()
        for _,_,subject in self.image_dictionary:
            subjects.add(subject)
        print ("loaded total image: %d, subject number is: %d" % (len(self.image_dictionary), len(subjects)))

        #self.get_eigen()
        #self.getWeight4Training()

    # use the method describing in Turk and Pentland's paper.
    def get_eigen(self):
        #计算平均脸
        mean_vector = self.vector_matrix.mean(axis=1)
        #计算每个样本与平均脸之间的差值向量
        for ii in range(self.vector_matrix.shape[1]):
            self.vector_matrix[:,ii] -= mean_vector
        shape = self.vector_matrix.shape
         # if there is huge number of training images. Usually go for 'else' branch.
        #判断图片维数（w*h）与训练图片数量大小
        if (shape[0]<shape[1]):
            #对差值矩阵进行奇异变换（相当于对差值矩阵的协方差矩阵A*AT求取特征值和特征向量）
            #简化计算A*AT与AT*A具有相同的非零特征值
            _,lamb, u = np.linalg.svd(np.dot(self.vector_matrix,self.vector_matrix.T))
            u = u.T
            # pass
        else:
            #若A的行数大于列数，则先计算AT*A减少计算量,再利用特征向量性质变换求取A*AT的特征向量
            _,lamb, v = np.linalg.svd(np.dot(self.vector_matrix.T, self.vector_matrix))
            v = v.T
            u = np.dot(self.vector_matrix,v)
            # Normalizing u to ||u||=1
            norm = np.linalg.norm(u,axis=0) #求取二范数进行单位化处理
            u = u / norm
            #lamb = lamb * norm  # don't need to do this normalized to eigenvalues
        # print (lamb)
        # print (np.linalg.norm(u,axis=0))
        standard_deviation = lamb**2/float(len(lamb))#计算特征值方差（对角矩阵）
        variance_proportion = standard_deviation / np.sum(standard_deviation)
        eigen = bunch.Bunch()
        eigen.lamb = lamb
        eigen.u = u
        eigen.variance_proportion = variance_proportion
        eigen.mean_vector = mean_vector
        self.eigen = eigen
        # The top K eigen value that represent 'most' of the variance in the training data
        self.K = self.get_number_of_components_to_preserve_variance(self.variance_pct)
        print ("get_number_of_components_to_preserve_variance: var=%.2f, K=%d" % (self.variance_pct,self.K))

    #计算使用PCA提取特征空间后特征脸权重
    def getWeight4Training(self):
        self.weightTraining = np.dot(self.eigen.u.T, self.vector_matrix)
        return self.weightTraining

    #计算特征向量权重
    def get_eigen_value_distribution(self):
        data = np.cumsum(self.eigen.lamb) / np.sum(self.eigen.lamb)
        return data

    #按重构阈值求取引入特征向量的个数
    def get_number_of_components_to_preserve_variance(self, variance=.95):
        for ii, eigen_value_cumsum in enumerate(self.get_eigen_value_distribution()):
            if eigen_value_cumsum >= variance:
                #print ("get_number_of_components_to_preserve_variance: var=%.2f, K=%d" % (variance,ii))
                return ii

    def getWeight4NormalizedImg(self, imgNormlized):
        return np.dot(self.eigen.u.T,imgNormlized)

    #计算图片在特征空间的权重
    def getWeight4img(self,img):
        return self.getWeight4NormalizedImg(img.flatten-self.eigen.mean_vector)

    #根据特征脸权重重构人脸图像
    def porject2eigenFaces(self, img,k=-1):
        if k<0:
            k = self.K
        ws = self.getWeight4NormalizedImg(img)
        u = self.eigen.u
        # imgNew = np.dot(u,ws)
        print("重构样本人脸所用特征脸个数"+str(k))
        imgNew = np.dot(self.eigen.u[:,0:k],ws[0:k])
        fig,axarr = plt.subplots(1,2)
        axarr[0].set_title(" porject2eigenFaces: original")
        axarr[0].imshow(img.reshape(self.imgShape) + self.get_average_weight_matrix(), cmap=plt.cm.gray)
        # axarr[0].imshow(img.reshape(self.imgShape) , cmap=plt.cm.gray)
        axarr[1].set_title(" porject2eigenFaces: projection")
        axarr[1].imshow(imgNew.reshape(self.imgShape) + self.get_average_weight_matrix(), cmap=plt.cm.gray)
        # axarr[1].imshow(imgNew.reshape(self.imgShape), cmap=plt.cm.gray)
        return imgNew

    # evaluate on training data using knn
    def eval(self, knn_k=-1,Kpca=-1):
        if knn_k <= 0:
            knn_k = self.knn
        knn_k += 1  # exclude itself
        if Kpca<0:
            Kpca = self.K

        responses = []
        for name,img,label in self.image_dictionary:
            responses.append(label)
        knn = cv2.ml.KNearest_create()
        knn.train(self.weightTraining[0:Kpca,:].T.astype(np.float32),cv2.ml.ROW_SAMPLE,np.asarray(responses,dtype=np.float32))
        # we have to discard the first predict result, since it has to be itself
        ret, results, neighbours2 ,dist = knn.findNearest(self.weightTraining[0:Kpca,:].T.astype(np.float32), knn_k)
        neighbours = neighbours2[:,1:]
        eval_data = []
        for idx,nb in enumerate(neighbours):
            neighbours_count = []
            for n in nb:
                neighbours_count.append(nb.tolist().count(n))
            vote = nb[neighbours_count.index(max(neighbours_count))]
            eval_data.append((vote,responses[idx]))
            # print ("predict:%s, neight: %s, label: %d" % (str(vote),str(nb), responses[idx]))
        return eval_data

    def get_eval(self, knn_k=-1,Kpca=-1):
        eval_data = self.eval(knn_k,Kpca)
        tp = 0
        fp = 0
        for pair in eval_data:
            if int(pair[0]) == int(pair[1]):
                tp += 1
            else:
                fp += 1
        precision = 1.0*tp/(tp+fp)
        return precision

    def get_testeval(self, e_data):
        tp = 0
        fp = 0
        for pair in e_data:
            if int(pair[0]) == int(pair[1]):
                tp += 1
            else:
                fp += 1
        precision = 1.0 * tp / (tp + fp)
        return precision


    ################   --------- show the result --------- ###################
    #显示训练集所有图片
    def plot_image_dictionary(self):
        dictionary = self.image_dictionary
        num_row_x = num_row_y = int(np.floor(np.sqrt(len(dictionary)-1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii, (name, v,_) in enumerate(dictionary):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(v, cmap=plt.cm.gray)
            axarr[div, rem].set_title('{}'.format(self.getClassFromName(name)).capitalize())
            axarr[div, rem].axis('off')
            if ii == len(dictionary) - 1:
                for jj in range(ii, num_row_x*num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

    #绘制特征脸
    def plot_eigen_vector(self, n_eigen=-1, nth=-1):
        if nth is -1:
            self.plot_eigen_vectors(n_eigen)
        else:
            plt.figure()
            plt.imshow(np.reshape(self.eigen.u[:,nth], self.imgShape), cmap=plt.cm.gray)

    # 绘制特征脸
    def plot_eigen_vectors(self,number=-1):
        if number<0:
            number = self.eigen.u.shape[1]
            print("特征个数"+number)
        num_row_x = num_row_y = int(np.floor(np.sqrt(number-1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii in range(number):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(np.reshape(self.eigen.u[:,ii], self.imgShape), cmap=plt.cm.gray)
            axarr[div, rem].set_title("%.6f" % self.eigen.variance_proportion[ii])
            axarr[div, rem].axis('off')
            if ii == number - 1:
                for jj in range(ii, num_row_x*num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

    def get_average_weight_matrix(self):
        return np.reshape(self.eigen.mean_vector, self.imgShape)

    def plot_mean_vector(self):
        fig,axarr = plt.subplots()
        axarr.set_title(" plot_mean_vector")
        axarr.imshow(self.get_average_weight_matrix(), cmap=plt.cm.gray)

    def plot_pca_components_proportions(self):
        fig,axarr = plt.subplots()
        plt.grid(True)
        plt.xlabel('number of components')
        plt.ylabel('Percentage of variance')
        axarr.set_title(" plot_pca_components_proportions")
        axarr.scatter(range(self.eigen.variance_proportion.size), self.eigen.variance_proportion)


    def plot_eigen_value_distribution(self):
        fig,axarr = plt.subplots()
        plt.grid(True)
        plt.xlabel('number of components')
        plt.ylabel('Percentage of variance')
        axarr.set_title(" plot_eigen_value_distribution")
        data = np.cumsum(self.eigen.lamb,) / np.sum(self.eigen.lamb)
        axarr.scatter(range(data.size), data)

    # plot the weights of top 2 component of all the training image
    def plotTrainingClass(self):
        fig,axarr = plt.subplots()
        axarr.set_title(" plotTrainingClass")
        ws = self.weightTraining
        axarr.scatter(ws[:,0],ws[:,1])
        for idx in range(0,ws.shape[0]):
            name = self.getClassFromName(self.image_dictionary[idx][0])
            axarr.text(ws[idx,0],ws[idx,1],name)
    def getClassFromName(self,fileName,lastSubdir=True):
        if lastSubdir:
            name = os.path.basename(os.path.dirname(fileName))
        else:
            name = os.path.basename(fileName)
        mat = re.match(".*(\d+).*", name)
        if mat != None:
            return int(mat.group(1))
        else:
            return name.__hash__()