import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import fowlkes_mallows_score

# 设置字体
myfont = fm.FontProperties(fname='ziti.TTF')
# 聚类评分
def get_best_cluster(x):
	scores = []
	for i in range(2, 7):
		# 构建并训练模型
		kmeans = KMeans(n_clusters = i, random_state=123).fit(x)
		score = fowlkes_mallows_score(iris['target'], kmeans.labels_) # 实际分类与预测分类比较，计算得分
		scores.append(score)
		print('iris数据聚{0}类FMI评价分值为：{1},SSE评分为:{2}'.format(str(i),str(score),str(kmeans.inertia_)))

	# 根据真实值评分，分3类的FMI评价最高，则最终非监督聚类3堆最合适
	max_score_index = scores.index(max(scores))
	K = max_score_index + 2
	print('分为:['+str(K)+"] 类最合适")
	return K

# 可视化显示
def disply_3d(X, model):
	labels = model.labels_
	fig = plt.figure("花聚类", figsize=(4, 3))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	np.random.seed(5)
	ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolor='k')
	ax.w_xaxis.set_ticklabels([0.1, 0.3, 0.5, 0.7, 0.9, 1.3, 1.5, 1.7])
	ax.w_yaxis.set_ticklabels([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
	ax.w_zaxis.set_ticklabels([1, 2, 3, 4, 5, 6, 7, 8])
	ax.set_xlabel('花瓣宽度', fontproperties=myfont)
	ax.set_ylabel('萼片长度', fontproperties=myfont)
	ax.set_zlabel('花瓣长度', fontproperties=myfont)
	ax.set_title("共"+str(set(labels))+"类", fontproperties=myfont)
	ax.dist = 12
	plt.show()

# 获取数据
iris = datasets.load_iris()
X = iris.data
K = get_best_cluster(X)

# 训练模型
model = KMeans(n_clusters=K)
model.fit(X)

# 预测数据
x_test = np.array([[4, 3, 1, 0.2]])
y_pre = model.predict(x_test)

# 打印结果
print("测试组数据:", x_test, " 属于第:", y_pre, "类")

# 显示数据
disply_3d(X,model)
