import eigenFace as ef
import matplotlib.pyplot as plt
import numpy as np


### for att face dataset
eigen_face = ef.EigenFace(variance_pct=0.99,knn=1)
eigen_face.get_eigen()
eigen_face.getWeight4Training()
### for yale face dataset
#eigen_face = EigenFace(variance_pct=0.99,knn=1,suffix="*[0-9].pgm")
eigen_face.plot_image_dictionary()
eigen_face.plot_eigen_vector(16)
eigen_face.plot_mean_vector()
eigen_face.plot_pca_components_proportions()
# eigen_face.plot_eigen_value_distribution()
eigen_face.plotTrainingClass()
eigen_face.K = eigen_face.get_number_of_components_to_preserve_variance(0.80)
print (eigen_face.K)
eigen_face.porject2eigenFaces(eigen_face.vector_matrix[:,21],35)

plt.figure()
plt.grid(True)
plt.xlabel('k neighbors in eigenfaces space')
plt.ylabel('Precision')
precisions = []
for i,var in enumerate([0.99,0.95,0.90,0.80,0.70,0.60,0.50]):
    precisions.append([])
    for j,knn_k in enumerate(range(1,10)):
        eigen_face.K = eigen_face.get_number_of_components_to_preserve_variance(var)
        pre = eigen_face.get_eval(knn_k)
        precisions[i].append(pre)
        print("knn_k: %2d, variance:%.2f(%d),\tprecision: %.4f" % (knn_k, var, eigen_face.K, pre))
    plt.plot(range(1,10),precisions[i],label="variance: %.0f%%" % (var*100),marker=i)
plt.legend(loc='best')

precisions = np.asarray(precisions)
print (precisions)
plt.show()