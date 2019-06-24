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

wordList = {'boy': [1, 1], 'like': [0, 2], 'girl': [3, 0],'car':[4, 4], 'auto':[5, 5], 'automobile':[6,0], 'machine':[0,7], 'motorcar':[8,8]}

# 感知门pGate,里面包含1*2个权重参数************************************************************************************
pGate_weight_text = tf.Variable(initial_value=tf.random_normal(shape=[1], mean=0, stddev=1), name='pGate_weight_text')
pGate_weight_voice = tf.Variable(initial_value=tf.random_normal(shape=[1], mean=0, stddev=1), name='pGate_weight_voice')
pGate = PerceptionGate.PerceptionGate(pGate_weight_text, pGate_weight_voice)
print('pGate感知门构建完成：')


def fusePGate( text, voice, weight_text, weight_voice):
    # 对两种表示加权后相连接，得到第一个门的链接向量表示
    w_text = tf.divide(weight_text, (weight_text + weight_voice))
    w_voice = tf.divide(weight_voice, (weight_text + weight_voice))
    representation4text = tf.multiply(w_text, text)
    representation4voice = tf.multiply(w_voice, voice)
    pgate_representation = tf.concat(values=[representation4text, representation4voice], axis=0)
    return pgate_representation





'''
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
'''

# 融合阶段
loss = tf.placeholder(dtype=tf.float32, name='loss')


#这里的参数暂时只写第一个门的参数，作为测试，后面的门的权重参数慢慢添加
def getWordRepresentation(word, pGate_weight_text, pGate_weight_voice):
    print(word)
    text = test4nltk.getTextRepresentation(word)
    voice = test4nltk.getVoiceRepresentation(word)  # 这个地方可能要构建一下词表示的tensor形式
    pGate_representation = fusePGate(text, voice, pGate_weight_text, pGate_weight_voice)
    #lpGate_representation = lpGate.fuse(word, text, voice)
    #ssGate_representation = ssGate.fuse(word, text, voice)
    word_representation = tf.divide(pGate_representation, 3)
    '''
    print('text类型：',type(text),
          '\nvoice类型：', type(voice),
          '\npGate_representation类型：', type(pGate_representation),
          '\nword_representation类型：', type(word_representation)
          )
    '''
    return word_representation


def error(word_representation, sword_representation):
    error = 0.0
    word_representation = getWordRepresentation(word)
    synsets = wn.synsets(word)
    print('------------------------------------------计算', word, '的损失 ')
    for synset in synsets:
        print(synset.lemma_names())
        for sword in synset.lemma_names():
            print('sword=', sword)
            if sword in wordList and sword != word:
                print('获取单词', sword, '的词表示')
                sword_representation = getWordRepresentation(sword)
#                print('获取单词', sword, '的词表示', sword_representation.eval())
                error += 1 - vectorOperations.cos_sim(word_representation, sword_representation)

    #print('单词', word, '的损失是', error.eval())
    return error


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # 将当前单词与所有synsets中的单词计算向量空间误差，在test4nltk中写函数
    # 每个单词对应三个门的表示在下面实现
    # 每次选1个词训练一下
    # 存储损失变量是loss

    step = 0
    for i in range(1000):
        step += 1
        for word in wordList:
            synsets = wn.synsets(word)
            word_representation = getWordRepresentation(word, pGate_weight_text, pGate_weight_voice)
            print('word_rep=', type(word_representation))
            print('------------------------------------------计算', word, '的损失 ')
            for synset in synsets:
                print(synset.lemma_names())
                for sword in synset.lemma_names():
                    print('sword=', sword)
                    if sword in wordList and sword != word:
                        print('获取单词', sword, '的词表示')
                        sword_representation = getWordRepresentation(sword)
                        # print('获取单词', sword, '的词表示', sword_representation.eval())
                        loss = 1 - vectorOperations.cos_sim(word_representation, sword_representation)
                train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
                sess.run(train_step, feed_dict={})
            # 选择的优化算法来不断地修改变量以降低损失:(下面是梯度下降算法)
            # 注意这里minimize（var_list）中的var_list必须含有梯度，也就是说必须是张量
            # 怎么修改呢，可以把获得单词表示的部分放在这里，然后用这些表示来①构建表示的张量形式②用这些张量来计算损失


            if step % 100 == 0:
                print('Step %d: loss = %.2f', step, error)


# print(lpGate_weight_text,lpGate_weight_voice)


# 隐感知门lpGate,里面包含41个参数,Jieba分词中可以分出41个词性



