import gensim
from numpy import *
word = 'boy'
#改进向量
f_Mmodel1 = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\Mmodel300.txt', 'r', encoding='utf-8')
Mdict0 = eval(f_Mmodel1.read())
print(word,len(Mdict0[word].tolist()),Mdict0[word].tolist())
f_Mmodel1 = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\Mmodel600.txt', 'r', encoding='utf-8')
Mdict1 = eval(f_Mmodel1.read())
print(word,len(Mdict1[word].tolist()),Mdict1[word].tolist())
f_Mmodel2 = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第三次实验\Mmodel800.txt', 'r', encoding='utf-8')
Mdict2 = eval(f_Mmodel2.read())
print(word,len(Mdict2[word].tolist()) ,Mdict2[word].tolist())
f_Mmodel3 = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第三次实验\Mmodel900.txt', 'r', encoding='utf-8')
Mdict3 = eval(f_Mmodel3.read())
print(word,len(Mdict3[word].tolist()),Mdict3[word].tolist())
#标准w2v向量
#model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\Administrator\Desktop\论文\数据\预训练词向量模型\GoogleNews-vectors-negative300.bin', binary=True)
print("---------------------------------资源加载完成----------------------------------\n")
print("---------------------------------输入查询单词----------------------------------\n")
word=input()
while word :
    print("word = ", word)
    #print(model[word].tolist(),"\n")
    print(Mdict1[word],"\n")
    print(Mdict2[word],"\n")
    print(Mdict3[word],"\n")
    word = input()
