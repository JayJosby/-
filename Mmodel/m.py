import tensorflow as tf
import random
from nltk.corpus import wordnet as wn


#训练数据处理部分,主要是文本/语音表示的获取和输入
# word.npy 存储着单词信息，包括词性信息。用列表格式读出，单位是word对象
# wordList.npy 存储着单词名称信息。用列表格式读出，单位是字符串，长度=23135
f = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\Edict.txt', 'r',encoding ='utf-8')    #英文单词总量20544
wordDict = eval(f.read())
fw = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\Mmodel.txt', 'w',encoding ='utf-8')    #英文单词总量20544


# f = open('/home/gpu/mylib/anaconda3/envs/liushuang/temp/Edict.txt', 'r',encoding ='utf-8')    #英文单词总量20544
# wordDict = eval(f.read())
# fw = open('/home/gpu/mylib/anaconda3/envs/liushuang/temp/Mmodel.txt', 'w',encoding ='utf-8')    #英文单词总量20544


def cosine(q, a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 0)+1e-8) #这个地方加一个极小值是为了避免导数为0
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 0)+1e-8)
    pooled_mul_12 = tf.reduce_sum(q * a, 0)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
    return score



# a = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\wordList.npy')
# wordList = a.tolist()

text = tf.placeholder(dtype=tf.float32, name='text_representation')
voice = tf.placeholder(dtype=tf.float32, name='voice_representation')
stext = tf.placeholder(dtype=tf.float32, name='stext_representation')
svoice = tf.placeholder(dtype=tf.float32, name='svoice_representation')
MyDict = {}

# 第一道感知门pGate,含有1*2个参数-----------------------------------------------------
pGate_weight_text = tf.Variable(initial_value=tf.random_normal(shape=[1], mean=0, stddev=1), name='pGate_weight_text')
pGate_weight_voice = tf.Variable(initial_value=tf.random_normal(shape=[1], mean=0, stddev=1), name='pGate_weight_voice')
pGate_weight_text = tf.divide(pGate_weight_text, (pGate_weight_text + pGate_weight_voice))
pGate_weight_voice = tf.divide(pGate_weight_voice, (pGate_weight_text + pGate_weight_voice))
# 获得word的所有表示，text + voice
pGate_text_word = tf.multiply(pGate_weight_text, text)
pGate_voice_word = tf.multiply(pGate_weight_voice, voice)
pGate_rep4word = tf.concat(values=[pGate_text_word, pGate_voice_word], axis=0)

# 获得sword的所有表示
pGate_text_sword = tf.multiply(pGate_weight_text, stext)
pGate_voice_sword = tf.multiply(pGate_weight_voice, svoice)
pGate_rep4sword = tf.concat(values=[pGate_text_sword, pGate_voice_sword], axis=0)

# 计算误差
pGate_loss = 1 - cosine(pGate_rep4word, pGate_rep4sword)

# 第二道门lpGate,含有35*2个参数--------------------------------------------------------
# 第一步，首先获得单词对应的词性频率列表
word_tagWeight = tf.placeholder(dtype=tf.float32)
sword_tagWeight = tf.placeholder(dtype=tf.float32)
# 下面的35个词性对应的text/voice之比是通用的
lpGate_weight_text = tf.Variable(initial_value=tf.random_normal(shape=[35], mean=0, stddev=1), name='lpGate_weight_text')
lpGate_weight_voice = tf.Variable(initial_value=tf.random_normal(shape=[35], mean=0, stddev=1), name='lpGate_weight_voice')
lpGate_weight_text = tf.divide(lpGate_weight_text, (lpGate_weight_text+lpGate_weight_voice))
lpGate_weight_voice = tf.divide(lpGate_weight_voice, (lpGate_weight_text+lpGate_weight_voice))

#获得word的所有表示信息
lpGate_weight_text_word = tf.multiply(word_tagWeight, lpGate_weight_text)
lpGate_weight_voice_word = tf.multiply(word_tagWeight, lpGate_weight_voice)
lpGate_text_word = tf.multiply(tf.reduce_sum(lpGate_weight_text_word), text)
lpGate_voice_word = tf.multiply(tf.reduce_sum(lpGate_weight_voice_word), voice)
lpGate_rep4word = tf.concat(values=[lpGate_text_word, lpGate_voice_word],axis=0)

#获得sword的所有表示信息
lpGate_weight_text_sword = tf.multiply(sword_tagWeight, lpGate_weight_text)
lpGate_weight_voice_sword = tf.multiply(sword_tagWeight, lpGate_weight_voice)
lpGate_text_sword = tf.multiply(tf.divide(tf.reduce_sum(lpGate_weight_text_sword), 35), stext)
lpGate_voice_sword = tf.multiply(tf.divide(tf.reduce_sum(lpGate_weight_voice_sword), 35), svoice)
lpGate_rep4sword = tf.concat(values=[lpGate_text_sword, lpGate_voice_sword],axis=0)

#lpGate的误差
lpGate_loss = 1 - cosine(lpGate_rep4word, lpGate_rep4sword)

# 第三道门lpGate,含有dict_length*2个参数--------------------------------------------------------
dict_length = 20544
index_word = tf.placeholder(dtype= tf.int32)
index_sword = tf.placeholder(dtype= tf.int32)

ssGate_weight_text = tf.Variable(initial_value=tf.random_normal(shape=[dict_length], mean=0, stddev=1), name='ssGate_weight_text')
ssGate_weight_voice = tf.Variable(initial_value=tf.random_normal(shape=[dict_length], mean=0, stddev=1), name='ssGate_weight_voice')
ssGate_weight_text = tf.divide(ssGate_weight_text, (ssGate_weight_text+ssGate_weight_voice))
ssGate_weight_voice = tf.divide(ssGate_weight_voice, (ssGate_weight_text+ssGate_weight_voice))

# 获得word的所有表示，text + voice，第一步获得当前单词的索引，然后根据这个索引找到对应的语音/文本权重
# 继续在下面的feed中添加index信息，如何使用索引print(sess.run(outputs[0:2,0:2,:]))
ssGate_text_word = tf.multiply(ssGate_weight_text[index_word], text)
ssGate_voice_word = tf.multiply(ssGate_weight_voice[index_word], voice)
ssGate_rep4word = tf.concat(values=[ssGate_text_word, ssGate_voice_word], axis=0)

# 获得sword的所有表示，text + voice
ssGate_text_sword = tf.multiply(ssGate_weight_text[index_sword], stext)
ssGate_voice_sword = tf.multiply(ssGate_weight_voice[index_sword], svoice)
ssGate_rep4sword = tf.concat(values=[ssGate_text_sword, ssGate_voice_sword], axis=0)
ssGate_loss = 1 - cosine(ssGate_rep4word, ssGate_rep4sword)


# 这一步是计算三个门机制融合后的词表示误差
rep4word = lpGate_rep4word + pGate_rep4word + ssGate_rep4word
loss = pGate_loss + lpGate_loss + ssGate_loss
saver = tf.train.Saver()



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    Los = 0
    for i in range(1000):
        # 下面的代码为了测试，改动了两处，一处是保证word 也在wordList_text中，另一处是保证sword在wordList_text中
        word_batch = random.sample(wordDict.keys(), 10)  # 随机一个字典中的key，第二个参数为限制个数，这里每次训练取10个单词作为训练Batch
        for word in word_batch :
            synsets = wn.synsets(word)
            for synset in synsets:
                if len(synset.lemma_names())!= 1:
                    for sword in synset.lemma_names():
                        if sword in wordDict:
                            step += 1
                            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
                            w_t = wordDict[word]['text']
                            w_v = wordDict[word]['voice']
                            sw_t = wordDict[sword]['text']
                            sw_v = wordDict[sword]['voice']
                            w_tagWeight = wordDict[word]['tag_weight']
                            sw_tagWeight = wordDict[sword]['tag_weight']
                            w_index = wordDict[word]['id']
                            sw_index = wordDict[sword]['id']
                            if len(w_t) != 300 or len(sw_t) != 300 or len(w_v)!=100 or len(sw_v)!=100 :
                                continue
                            _, los, rep = sess.run([train_step, loss, rep4word], feed_dict={text: w_t, voice: w_v, stext: sw_t, svoice: sw_v, word_tagWeight:w_tagWeight, sword_tagWeight:sw_tagWeight, index_word:w_index, index_sword:sw_index})
                            print(los)
                            MyDict[word] = rep
                            Los += los
        if step % 2 == 0:
            Los /= step
            print('打印损失：', Los)
            fw.write(str(MyDict))
            Los = 0

        #saver.save(sess, save_path='/home/gpu/mylib/anaconda3/envs/liushuang/temp/jojo',global_step=5)
        saver.save(sess, save_path='C:\\Users\Administrator\Desktop\论文\数据\词性信息/jojo',global_step=5)



#
#
#
#
#
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     # 将当前单词与所有synsets中的单词计算向量空间误差，在test4nltk中写函数
#     # 每个单词对应三个门的表示在下面实现
#     # 每次选1个词训练一下
#     step = 0
#     for i in range(1000):
#         step += 1
#         for word in wordList:
#             word2 = word.word
#             synsets = wn.synsets(word2)
#             word_representation = getWordRepresentation(word2, pGate_weight_text, pGate_weight_voice)
#             print('word_rep=', type(word_representation))
#             print('------------------------------------------计算', word2, '的损失 ')
#             for synset in synsets:
#                 print(synset.lemma_names())
#                 for sword in synset.lemma_names():
#                     print('sword=', sword)
#                     if sword in wordList and sword != word2:
#                         print('获取单词', sword, '的词表示')
#                         sword_representation = getWordRepresentation(sword)
#                         # print('获取单词', sword, '的词表示', sword_representation.eval())
#                         error += 1 - vectorOperations.cos_sim(pGate_weight_text, pGate_weight_voice)
#                 train_step = tf.train.GradientDescentOptimizer(0.01).minimize(error)
#                 sess.run(train_step)
#     for i in range(1000):
#         step += 1
0#         for word in wordList:
#             synsets = wn.synsets(word)
#             word_representation = getWordRepresentation(word, pGate_weight_text, pGate_weight_voice)
#             print('word_rep=', type(word_representation))
#             print('------------------------------------------计算', word, '的损失 ')
#             for synset in synsets:
#                 print(synset.lemma_names())
#                 for sword in synset.lemma_names():
#                     print('sword=', sword)
#                     if sword in wordList and sword != word:
#                         print('获取单词', sword, '的词表示')
#                         sword_representation = getWordRepresentation(sword)
#                         # print('获取单词', sword, '的词表示', sword_representation.eval())
#                         error += 1 - vectorOperations.cos_sim(pGate_weight_text, pGate_weight_voice)
#                 train_step = tf.train.GradientDescentOptimizer(0.01).minimize(error)
#                 sess.run(train_step)
#             # 选择的优化算法来不断地修改变量以降低损失:(下面是梯度下降算法)
#             # 注意这里minimize（var_list）中的var_list必须含有梯度，也就是说必须是张量
#             # 怎么修改呢，可以把获得单词表示的部分放在这里，然后用这些表示来①构建表示的张量形式②用这些张量来计算损失
#
#
#             if step % 100 == 0:
#                 print('Step %d: loss = %.2f', step, error)


# print(lpGate_weight_text,lpGate_weight_voice)


# 隐感知门lpGate,里面包含41个参数,Jieba分词中可以分出41个词性



