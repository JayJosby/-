import gensim
import numpy as np
from math import sqrt
from numpy import *
import pandas as pd
#首先构造三个(当然不止3个)dict：dict dictW2V dictM
#以下代码可以实现构造最初的人工标注的数据集的dict,3个dict最好同时构造，先保存下来，因为w2v的载入太耗费时间了
#在没有得到dictM之前，先试试dict和dictW2V之间的计算，
def initDict():
    f = open('C:\\Users\Administrator\Desktop\论文\数据\测试数据\wordsim\en\EN-MC-30.txt', 'r',encoding ='utf-8')    #英文单词总量20544
    f_Mmodel = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\Edict3.txt', 'r',encoding ='utf-8')
    #f_Mmodel = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\ws353\第二次结果\Mmodel.txt', 'r', encoding='utf-8')
    dict_Mmodel = eval(f_Mmodel.read())
    model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\Administrator\Desktop\论文\数据\预训练词向量模型\GoogleNews-vectors-negative300.bin', binary=True)
    # 打开全规模字典文件
    print('已加载W2V文件')
    # print('准备开始构造dict_Mmodel')
    # print(type(dict_Mmodel))
    # for w in dict_Mmodel.keys():
    #     print(w)
    #     list_M = []
    #     text = dict_Mmodel[w]
    #     print(text)
    #     voice = dict_Mmodel[w]['voice']
    #     dict_Mmodel[w] = text + voice
    # f_save.write(str(dict_Mmodel))
    # print('dict_Mmodel构造完毕')
    #存储匹配ws353规模的dict数据
    f_dict = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\dict.txt', 'w',encoding ='utf-8')
    f_dictW2V = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\dictW2V.txt', 'w',encoding ='utf-8')
    f_dictM = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\dictM.txt', 'w',encoding ='utf-8')

    line = f.readline()
    dict = []
    dictW2V = []
    dictM = []
    #key in dict
    print('开始构造list')
    mark = 0
    while line:
        w=line.split()
        if w[0] in dict_Mmodel and w[1] in dict_Mmodel:
        #     # dict 构造
            l = [w[0], w[1], round(float(w[2])/10,2)]
            dict.append(l)
            #dictW2V构造
            lW2V = [w[0], w[1], round(model.similarity(w[0],w[1]), 2)]
            dictW2V.append(lW2V)
            #dictM构造
            lM = [w[0], w[1], round(cos_sim(np.array(dict_Mmodel[w[0]]),np.array(dict_Mmodel[w[1]])), 2)]
            dictM.append(lM)

        else:
            mark += 1
            print(w[0],w[1],"不在Edict3中")
        line = f.readline()
    print('单词对数量=',mark)
    #以下代码实现dict文件的保存
    f_dict.write(str(dict))
    f_dictW2V.write(str(dictW2V))
    f_dictM.write(str(dictM))
    f_dict.close()
    f_dictW2V.close()
    f_dictM.close()
    # 以下代码可以实现构造基于W2V的dict



#实现一个dict转list函数
# f_dict = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\dict.txt', 'r',encoding ='utf-8')
# f_dictW2V = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\dictW2V.txt', 'r',encoding ='utf-8')
# dict = eval(f_dict.read())
# dictW2V = eval(f_dictW2V.read())

def cos_sim(x, y):
    score = 0
    score = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return score

def d2l(d):
    l = []
    for w in d:
        w = d[w]
        l.append(w[1])
    print(l)
    return l


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den
#print(corrcoef(x, y))  # 0.471404520791

if __name__ == '__main__':
    initDict()
    f_dict = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\\dict.txt', 'r',encoding ='utf-8')
    f_dictW2V = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\\dictW2V.txt', 'r',encoding ='utf-8')
    f_dictM = open('C:\\Users\Administrator\Desktop\论文\数据\训练数据\\dictM.txt', 'r',encoding ='utf-8')

    l = eval(f_dict.read())
    lW2V =eval(f_dictW2V.read())
    lM = eval(f_dictM.read())

    for i in range(len(l)):
        l[i] = l[i][2]
    for i in range(len(lW2V)):
        lW2V[i] = lW2V[i][2]
    for i in range(len(lM)):
        lM[i] = lM[i][2]

    df = pd.DataFrame({'ws353': l, 'w2v': lW2V, 'Mmodel': lM})
    print(df.corr())
    print(df.corr('spearman'))
    print(df.corr('kendall'))
