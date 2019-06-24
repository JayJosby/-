import tensorflow as tf
import numpy as np
from test import test4nltk
from Mmodel import PerceptionGate
from Mmodel import LatentPerceptionGate
from Mmodel import SampleSpecificGate
from nltk.corpus import wordnet as wn
from test import test4nltk
from test import vectorOperations

#训练数据处理部分,主要是文本/语音表示的获取和输入
text = tf.placeholder(tf.float32, shape=[None, ])
voice = tf.placeholder(tf.float32, shape=[None, ])
text_test = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
voice_test = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
wordList = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy')
wordList = wordList.tolist()

# 感知门pGate,里面包含1*2个权重参数************************************************************************************
pGate_weight_text = tf.Variable(initial_value=tf.random_normal(shape=[1], mean=0, stddev=1), name='pGate_weight_text')
pGate_weight_voice = tf.Variable(initial_value=tf.random_normal(shape=[1], mean=0, stddev=1), name='pGate_weight_voice')
pGate = PerceptionGate.PerceptionGate(pGate_weight_text, pGate_weight_voice)
print('pGate感知门构建完成：')

# 隐感知门lpGate,里面包含35*2个权重参数********************************************************************************
lpGate_weight_text = tf.Variable(initial_value=tf.random_normal(shape=[35], mean=0, stddev=1), name='plGate_weight_text')
lpGate_weight_voice = tf.Variable(initial_value=tf.random_normal(shape=[35], mean=0, stddev=1), name='plGate_weight_voice')
#获得当前词text
lpGate = LatentPerceptionGate.LantentPerceptionGate(lpGate_weight_text, lpGate_weight_voice)
print('lpGate隐感知门构建完成,阶段表示：')

#词抽样门ssGate,里面包含维度==词典维度的权重,当然作为目标函数中的输入数量要少很多，因为很多单词在wordnet中都无法判断关系，比如奔驰Benz就查不到，而奔驰在测试集中可能是存在的
ssGate_weight_text = tf.Variable(initial_value=tf.random_normal(shape=[23135], mean=0, stddev=1), name='ssGate_weight_text')
ssGate_weight_voice = tf.Variable(initial_value=tf.random_normal(shape=[23135], mean=0, stddev=1), name='ssGate_weight_voice')
ssGate = SampleSpecificGate.SampleSpecificGate(ssGate_weight_text, ssGate_weight_voice)
ssGate_representation = ssGate.fuse(text, voice)
print('ssGate词采样们构建完成，阶段表示：')


def getWordRepresentation(word):
    text = test4nltk.getTextRepresentation(word)
    voice = test4nltk.getVoiceRepresentation(word)  # 这个地方可能要构建一下词表示的tensor形式
    pGate_representation = pGate.fuse(text, voice)
    lpGate_representation = lpGate.fuse(word, text, voice)
    ssGate_representation = ssGate.fuse(word, text, voice)
    word_representation = tf.divide(pGate_representation + lpGate_representation + ssGate_representation, 3)
    return word_representation


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # 将当前单词与所有synsets中的单词计算向量空间误差，在test4nltk中写函数
    # 每个单词对应三个门的表示在下面实现
    # 每次选1个词训练一下

    error = tf.placeholder([0])
    word_representation = getWordRepresentation(word)
    synsets = wn.synsets(word)
    for synset in synsets:
        for sword in synset.lemma_names:
            if sword in wordList:
                sword_representation = getWordRepresentation(sword)
                error += 1 - vectorOperations.cos_sim(word_representation, sword_representation)
    cross_entropy = tf.reduce_sum(error)
    # 选择的优化算法来不断地修改变量以降低损失:(下面是梯度下降算法)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    for i in range(1000):
        for word in wordList:
            sess.run(train_step, feed_dict={word: word})


# print(lpGate_weight_text,lpGate_weight_voice)


# 隐感知门lpGate,里面包含41个参数,Jieba分词中可以分出41个词性



