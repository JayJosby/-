import gensim
import numpy as np
from math import sqrt
from numpy import *
import pandas as pd
import os
#首先构造二个(当然不止3个)dict：dict dictW2V dictM
#以下代码可以实现构造最初的人工标注的数据集的dict,3个dict最好同时构造，先保存下来，因为w2v的载入太耗费时间了
#在没有得到dictM之前，先试试dict和dictW2V之间的计算，

#dataset = 'EN-WS-353-ALL'
dataset = 'EN-WS-353-REL'
#dataset = 'EN-WS-353-SIM'
#dataset = 'EN-RG-65'
#dataset = 'EN-MC-30'
#dataset = 'EN-YP-130'
#dataset = 'EN-MTurk-287'
#dataset = 'EN-MTurk-771'
#dataset = 'EN-RW-STANFORD'
#dataset = 'EN-MEN-TR-3K'

#正常路径是
dir = ''

file = dir + '\\res_600_2_no_ss.txt'
fileW2V = dir + '\\resW2V_600_2_no_ss.txt'
fileM = dir + '\\resM_600_2_no_ss.txt'

newmarkfile = dir + '\\mark_600_2_no_ss.txt'

num = '2'

wordnum = 0

def initDict(dataset):
    f = open('C:\\Users\Administrator\Desktop\论文\数据\测试数据\wordsim\en\\'+dataset+'.txt', 'r', encoding='utf-8')
    f_Mmodel = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\Mmodel600_2_no_ss.txt', 'r', encoding='utf-8')
    dict_Mmodel = eval(f_Mmodel.read())
    model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\Administrator\Desktop\论文\数据\预训练词向量模型\GoogleNews-vectors-negative300.bin', binary=True)
    # 打开全规模字典文件
    print('已加载W2V文件')

    #存储匹配ws353规模的dict数据
    isDir('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\'+dataset+'\\'+dir)
    f_dict = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\'+dataset + file, 'w',encoding ='utf-8')
    f_dictW2V = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\'+dataset + fileW2V, 'w',encoding ='utf-8')
    f_dictM = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\'+dataset + fileM, 'w',encoding ='utf-8')

    line = f.readline()
    dict = []
    dictW2V = []
    dictM = []
    #key in dict
    print('开始构造list')
    while line:
        w=line.split()
        if w[0] in dict_Mmodel and w[1] in dict_Mmodel and len(dict_Mmodel[w[0]]) == len(dict_Mmodel[w[1]]):
            r_DataSet = round(float(w[2])/10,2)
            r_W2V = round(model.similarity(w[0],w[1]), 2)
            r_Mmodel = round(cos_sim(np.array(dict_Mmodel[w[0]]),np.array(dict_Mmodel[w[1]])),2)

            diff_dict_dictW2V = abs(r_W2V-r_DataSet)
            diff_dictW2V_dictM = abs(r_Mmodel-r_DataSet)

            if diff_dict_dictW2V >= diff_dictW2V_dictM:
                dict.append([w[0],w[1],r_DataSet])
                dictW2V.append([w[0],w[1],r_W2V])
                dictM.append([w[0],w[1],r_Mmodel])
                print(r_DataSet, r_W2V, r_Mmodel)
            else:
                #if random.randint(0,10,1) >8:
                print('-------------------------------------',r_DataSet, r_W2V, r_Mmodel,'')
                dict.append([w[0], w[1], r_DataSet])
                dictW2V.append([w[0], w[1], r_W2V])
                dictM.append([w[0], w[1], r_Mmodel])
            #print(w[0],w[1],"不在Edict3中（或者是被滤掉）")
        line = f.readline()
    #以下代码实现dict文件的保存
    f_dict.write(str(dict))
    f_dictW2V.write(str(dictW2V))
    f_dictM.write(str(dictM))
    f_dict.close()
    f_dictW2V.close()
    f_dictM.close()

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

def isDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


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

def getCorrcoef(dataset):
    f_dict = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\' + dataset  + file, 'r',
                  encoding='utf-8')
    f_dictW2V = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\' + dataset + fileW2V, 'r',
                     encoding='utf-8')
    f_dictM = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\' + dataset  +  fileM, 'r',
                   encoding='utf-8')

    l = eval(f_dict.read())
    lW2V = eval(f_dictW2V.read())
    lM = eval(f_dictM.read())

    for i in range(len(l)):
        l[i] = l[i][2]
    for i in range(len(lW2V)):
        lW2V[i] = lW2V[i][2]
    for i in range(len(lM)):
        lM[i] = lM[i][2]

    df = pd.DataFrame({'ws353': l, dataset: lW2V, 'Mmodel': lM})
    print("当前数据集为:",dataset,' 当前是第',num,'次实验')
    print('皮尔逊相关系数=\n',df.corr())
    print('斯皮尔曼相关系数=\n',df.corr('spearman'))
    print('肯德尔相关系数=\n',df.corr('kendall'))

    f = open('C:\\Users\Administrator\Desktop\论文\任务记录\词相似度\英文\第二次实验\\'+ dataset + newmarkfile, 'w', encoding='utf-8')
    mark = "当前数据集为:"+str(dataset)+' 当前是第'+str(num)+'次实验\n'+'皮尔逊相关系数=\n'+str(df.corr())+'\n斯皮尔曼相关系数=\n'+str(df.corr('spearman'))+'\n肯德尔相关系数=\n'+str(df.corr('kendall'))+"\n当前词对数量="+str(wordnum)
    f.write(str(mark))
    f.close()


if __name__ == '__main__':
    datalist=['EN-WS-353-ALL','EN-WS-353-REL','EN-WS-353-SIM','EN-RG-65','EN-MC-30','EN-YP-130', 'EN-MTurk-287','EN-MTurk-771', 'EN-RW-STANFORD','EN-MEN-TR-3K']
    for data in datalist:
        print('开始数据集',data,'的测试')
        initDict(data)
        getCorrcoef(data)
