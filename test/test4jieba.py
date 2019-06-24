import jieba
import jieba.posseg as pseg

# -*- coding: utf-8 -*-

import jieba
import jieba.analyse
import nltk
# -*- coding: utf-8 -*-

import jieba
import jieba.analyse
import jieba.posseg as pseg
def getIndexOfCharacteristic(text):
    characteristics=pseg.cut(text)
    for characteristic in characteristics:
        print(characteristic)



text_test='dog'
getIndexOfCharacteristic(text_test)
#-----------------------------------------------------


def getWordCharacteristic(text):
    words = nltk.word_tokenize(text)
    print(words)
    tag = nltk.pos_tag(words)
    print(tag)


word = 'fuck'
print(getWordCharacteristic(word))

'''
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

with open('C:\\Users\Administrator\Desktop\论文\技术预备\第一阶段/in_the_name_of_people.txt',encoding='utf-8') as f:
    document = f.read()

    # document_decode = document.decode('GBK')

    document_cut = jieba.cut(document)
    # print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('C:\\Users\Administrator\Desktop\论文\技术预备\第一阶段/in_the_name_of_people_s.txt', 'w',encoding='utf-8') as f2:
        result=result.decode('utf-8')
        f2.write(result)
f.close()
f2.close()
'''

