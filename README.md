# surf_seimese_similarity
工程名:surf_seimese_similarity
工程作用：利用surf进行图像配准,再根据seimese孪生网络模型输出配准后的测试样本的相似度
工程目录：
	文件夹71:测试样本集合目录
	model:孪生网络模型输出目录
	model目录：
		bayes-0020.params(bayes是模型前缀名，0020表示模型迭代20次输出，params代表模型文件)
		bayes-symbol.json(网络结构文件)
	
	ven:程序运行的环境，该环境并没有搭建完成，在在本地我用的是anaconda的python编译环境，包含mxnet和opencv以及opencv_contrib。
		如果是内网机，离线安装contrib，需要下载3.4.3版本下的python-opencv-contirbe,否则无法在内网安装
	
	脚本py文件：
		triplet_loss.py是孪生网络训练脚本，输出即是model目录下的文件
		python_surf.py是surf进行配准的脚本文件
		computer_similarity.py是调用python_surf进行配准并计算相似度的脚本文件
		myTest.py是测试文件，最后输出一个dict(),value是包含了与基准图最相似的图片样本

