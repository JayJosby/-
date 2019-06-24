'''
展平图片的数字数组会丢失图片的二维结构信息。这显然是不理想的，最优秀的计算机视觉方法会挖掘并利用这些结构信息；
图片集60000*784
标签集60000*10

先引入数据:
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
里面含有mnist.train训练+mnist.test测试两部分

如何表示张量？
(用占位符)
x = tf.placeholder("float",[None,784])
(用Variable,比占位符方便，可在计算中修改，一般模型参数(权值之类)都用这个表示)
x = tf.Variable(tf.zeros([None,784]))

y = tf.nn.softmax(tf.matmul(x,w)+b)
(这里的softmax有什么用？可以看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们想要的格式，也就是关于10个数字类的概率分布。因为最后得到的y向量是各个选项的概率，所以根据容易比值得出)
损失函数定义:
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
选择的优化算法来不断地修改变量以降低损失:
(下面是梯度下降算法)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
(TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。)

基本上就是这样了，但是这个架构还需要其他一些操作:
所有的变量初始化:
init = tf.initialize_all_variables()
在Session中启动模型:
sess = tf.Session()
sess.run(init)
开始训练模型:
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)              //随机抓取训练数据中的100个批处理数据点
  sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
(这步为什么每次选100个数据？使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。)

用到的函数：
tf.reduce_sum(arg1,arg2),这个函数可以对矩阵降维，arg2取值0(压缩成一行)/1(压缩成一列)，reduce_后面出了sum方法还可以换成mean等方法。
tf.argmax(x，n)能给出的对象x的某一维最大值的索引。
tf.cast（x，type A),将对象x转换为A类型
tf.reshape(x,shape),将对象尺寸修改，可改变维度，卷积时常用图像1->4
eval()可以理解tf.Tensor的Session.run() 的另外一种写法，
	accuracy.eval({x:mnist.test.images,y_: mnist.test.labels})相当于
	sess.run(accuracy, {x:mnist.test.images,y_: mnist.test.labels})
-------------------------------------------------------------------------
什么是全连接层？
卷积神经网路中的全连接层。
在卷积神经网络中全连接层位于网络模型的最后部分，负责对网络最终输出的特征进行分类预测，得出分类结果。
池化层有什么用？
对输入的特征图进行压缩，一方面使特征图变小，简化网络计算复杂度；一方面进行特征压缩，提取主要特征。
全连接层有什么用？
全连接层的每一个结点都与上一层的所有结点相连，用来把前边提取到的特征综合起来。由于其全相连的特性，一般全连接层的参数也是最多的
------------------------------------------------------------------------
感觉把数据维度理解为特征数量比较直接，比数学概念容易想

卷积相关:
卷积为何能提取特征？
filter/卷积核/过滤器*特定特征矩阵=较大值
-------------------------------------------------------------
sigmoid:Sigmoid函数是一个在生物学中常见的S型函数，也称为S型生长曲线。在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间。
-------------------------------------------------------------
tf.nn.conv2d(input, filter(卷积核), strides(步幅), padding(填充), use_cudnn_on_gpu=None, name=None)
inpute的四个维度是[batch, in_height, in_width, in_channels]，
filter的四个维度是[filter_height, filter_width, in_channels, out_channels]//filter的输入通道与input的输入通道是一致的，而输出通道数就是filter的数量(即特征数)
-------------------------------------------------------------
①卷积核大小（Kernel Size）:卷积核大小定义了卷积的视野。2维中的常见选择是3 - 即3x3像素矩阵。
②步长（Stride）:步长定义遍历图像时卷积核的移动的步长。虽然它的默认值通常为[1, height, width, 1]，前后固定为1，第二/三个是水平/垂直滑动步长。
③填充（Padding）:填充定义如何处理样本的边界。有两种valid(边界不足卷积就丢弃) same(为了运算，边界补0，尺寸仍可能变化)
④输入和输出通道（Channels）:卷积层通常需要一定数量的输入通道（I），并计算一定数量的输出通道（O）。可以通过I * O * K来计算所需的参数，其中K等于卷积核中参数的数量，即卷积核大小。
-------------------------------------------------------------
池化相关:
tf.nn.max_pool(value, ksize, strides, padding, name=None)
ksize池化窗口的大小，一般是[1, height, width, 1]

如何保存模型?------------------------------------------------
saver=tf.train.Saver()//写在变量初始化函数之后

saver.save(sess,'path',global_step=i)//在for i in range(2000)

如何恢复模型?------------------------------------------------
saver.restore(sess,'path')//写在整个模型图定义完成之后



tensorflow程序保存生成四个文件:
①checkpointer			    文本文件，记录模型文件路径信息列表
②model.ckpt.data-00000-of-00001    网络权重信息
③model.ckpt.data和model.ckpt.index 二进制文件，保存变量参数(权重)信息
④model.ckpt.meta 		    二进制文件，保存模型计算图结构信息

tensorflow设置GPU显存:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)#分配内存大小
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

------------------------------------------------------------
当GPU不可使用时，切换到CPU:
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(graph=train_model.graph, config=config)





'''
''' 
import tensorflow as tf
import nltk
from nltk.corpus import wordnet as wn

print(wn.synsets('CAR'))
print(wn.synset('car.n.01').lemma_names())  #该词组下包含哪些单词
print(wn.synset('car.n.01').definition())   #该词组定义
print(wn.synset('car.n.01').examples())     #包含该含义的例句

a=tf.constant([2.0,4.0,6.0])
b=tf.constant([2.0])
c=tf.divide(a,b)
print(tf.divide(a,b))

with tf.Session() as sess:
    print(sess.run(c))
'''
import re
import numpy as np
from nltk.corpus import wordnet as wn
import tensorflow as tf

a = np.arange(12)
a = a.reshape(3,4)
for i in range(3):
    t = sum(a[i])
    for j in range(4):
        a[i,j] = a[i,j]/t
