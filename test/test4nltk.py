from nltk.corpus import wordnet as wn
import nltk
import numpy as np
import gensim
from test import word
from test import vectorOperations
import re



def getWordCharacteristic(text):
    words = nltk.word_tokenize(text)
    print(words)
    tag = nltk.pos_tag(words)
    print(tag)


def saveWordCharacteristic(textWithCharacteristic):
    # 保存
    for word in textWithCharacteristic:
        dict_name = {1 :{1 :2 ,3 :4} ,2 :{3 :4 ,4 :5}}
        f = open('C:\\Users\Administrator\Desktop\\temp.txt', 'w')
        f.write(str(dict_name))
        f.close()


def readWordCharacteristic():
    # 读取
    f = open('C:\\Users\Administrator\Desktop\\temp.txt' , 'r')
    a = f.read()
    dict_name = eval(a)
    print(dict_name)
    f.close()


def showWordList(wList):
    for w in wList:
        print('单词:', w.word)
        print('共出现次数:', w.sum)
        print('CC', w.freCC)
        print('CD', w.freCD)
        print('DT', w.freDT)
        print('EX', w.freEX)
        print('FW', w.freFW)
        print('IN', w.freIN)
        print('JJ', w.freJJ)
        print('JJR', w.freJJR)
        print('JJS', w.freJJS)
        print('LS', w.freLS)
        print('KMD', w.freKMD)
        print('NN', w.freNN)
        print('NNS', w.freNNS)
        print('NNP', w.freNNP)
        print('NNPS', w.freNNPS)
        print('PDT', w.frePDT)
        print('PRP', w.frePRP)
        print('RP', w.freRP)
        print('SYM', w.freSYM)
        print('VB', w.freVB)
        print('VBD', w.freVBD)
        print('VBG', w.freVBG)
        print('VBN', w.freVBN)
        print('VBP', w.freVBP)
        print('VBZ', w.freVBZ)
        print('WDT', w.freWDT)
        print('WP', w.freWP)
        print('WP$', w.freWPS)
        print('WRB', w.freWRB)
        print('TO', w.freTO)
        print('UH', w.freUH)
        print('RPR$', w.frePRPS)
        print('RB', w.freRB)
        print('RBR', w.freRBR)
        print('RBS', w.freRBS)
        print('--------------------------------')


def initWordList(wList):
    wList = []  # 将保存数据清除
    wList = np.array(wList)
    showWordList(wList)
    np.save('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy', wList)
    word_test = word.word()
    word_test.setWord('1', 'CC')
    wList.append(word_test)


def getWordListLength(wList):
    sum = 0
    for w in wList:
        sum += 1
    return sum


def getWordTagWeight(wList, word):
    mark = 0
    for w in wList:
        if word == w.word:
            charaList = [w.freCC
                        ,w.freCD
                        ,w.freDT
                        ,w.freEX
                        ,w.freFW
                        ,w.freIN
                        ,w.freJJ
                        ,w.freJJR
                        ,w.freJJS
                        ,w.freLS
                        ,w.freKMD
                        ,w.freNN
                        ,w.freNNS
                        ,w.freNNP
                        ,w.freNNPS
                        ,w.frePDT
                        ,w.frePRP
                        ,w.frePRPS
                        ,w.freRP
                        ,w.freSYM
                        ,w.freVB
                        ,w.freVBD
                        ,w.freVBG
                        ,w.freVBN
                        ,w.freVBP
                        ,w.freVBZ
                        ,w.freWDT
                        ,w.freWP
                        ,w.freWPS
                        ,w.freWRB
                        ,w.freTO
                        ,w.freUH
                        ,w.freRB
                        ,w.freRBR
                        ,w.freRBS]
            mark = 1
            break
        else:
            # print('词性文件中无此单词:', word)
            charaList = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
    return charaList


def getTextRepresentation(word):
    wordList = {'boy': [1, 1, 1, 1, 1], 'like': [2, 2, 2, 2, 2], 'girl': [3, 3, 3, 3, 3], 'car': [4, 4, 4, 4, 4],
                'auto': [5, 5, 5, 5, 5], 'automobile': [6, 6, 6, 6, 6], 'machine': [7, 7, 7, 7, 7],
                'motorcar': [8, 8, 8, 8, 8, ]}

    return wordList[word]


def getVoiceRepresentation(word):
    wordList = {'boy': [1, 1, 1, 1, 1], 'like': [2, 2, 2, 2, 2], 'girl': [3, 3, 3, 3, 3], 'car': [4, 4, 4, 4, 4],
                'auto': [5, 5, 5, 5, 5], 'automobile': [6, 6, 6, 6, 6], 'machine': [7, 7, 7, 7, 7],
                'motorcar': [8, 8, 8, 8, 8, ]}

    return wordList[word]



def test():
    a = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\wordList.npy')
    wordList = a.tolist()
    mark = 0
    print(wordList)
    list = []
    for ww in wordList:
        an = re.search('^[a-z]+$', ww)
        if an:
            mark += 1
            list.append(ww)
        else:
            print(ww)

    print(list)
    list = np.array(list)
    np.save('C:\\Users\Administrator\Desktop\论文\数据\词性信息\wordList.npy', list)

    '''
      list = []
    for w in wordList:
        list.append(w.word)
    list = np.array(list)
    np.save('C:\\Users\Administrator\Desktop\论文\数据\词性信息\wordList.npy', list)
    '''

    print('单词列表存储完成,词数',mark)
    return None


def isWordInList(wList , word):
    mark = '没有该单词'
    for w in wList:
        if word == w.word:
            mark = '包含该单词'+word
            break
    return mark

def setTagWeight():
    f = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\Edict.txt', mode='r',encoding='utf-8')
    fw = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\Edict3.txt', mode='w', encoding='utf-8')
    dict = eval(f.read())

    for word in dict:
        print(word)
        tag = [0, 0, 0, 0, 0]
        for key in wn.synsets(word):
            if key.pos() == 'a':
                tag[0] = 1
            elif key.pos() == 's':
                tag[1] = 1
            elif key.pos() == 'r':
                tag[2] = 1
            elif key.pos() == 'n':
                tag[3] = 1
            elif key.pos() == 'v':
                tag[4] = 1
            print("=",tag,"\n")
            dict[word]['tag_weight'] = tag
    fw.write(str(dict))
    print("tag_weight初始化完成")


def initSoundMark():
    f = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\test3.txt', mode='r',encoding='utf-8')
    fw = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\Edict.txt', mode='w',encoding='utf-8')
    #使用W2V谷歌新闻语料来充当text维度
    model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\Administrator\Desktop\论文\数据\预训练词向量模型\GoogleNews-vectors-negative300.bin', binary=True)
    a = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy')
    wordList = a.tolist()
    line1= f.readline().replace('\n','').replace('\ufeff','')
    step = 0
    dict = {}
    sub_dict = {}
    soundmarkLen = 0
    while line1:
        line2 = f.readline()#音标行
        line3 = f.readline()#空行
        if line1 != line2:
            step += 1
            soundmarkvector = getSoundmarkVector(line2) #嵌套词典
            sub_dict = {}
            sub_dict['soundmark'] = line2.replace('\n','')
            sub_dict['tag_weight'] = getWordTagWeight(wordList,line1)          #词性权重设置
            #sub_dict['text'] = getTextVector(model, line1)                    #词向量设置
            sub_dict['text'] = getTextVector(model, line1)                     #词向量设置
            print(line1+'\n',getTextVector(model, line1))
            sub_dict['voice'] = soundmarkvector                                 #声音向量设置
            sub_dict['id'] = step
            dict[line1] = sub_dict
        line1 = f.readline().replace('\n','')
        if line1 == ' \n' or line2 == ' \n':
            exit()
    fw.write(str(dict))
    print('有音标单词总数为',step)
    f.close()
    fw.close()

def getTextVector(model, text):
    if text in model:
        return model[text].tolist()
    else:
        return [1]*100


def getSoundmarkVector(soundmark):
    length = len(soundmark)-1
    if '(' in soundmark:
        length -= 1
    if ')' in soundmark:
        length -= 1
    length_loss = 15 - length
    turn = int(100/length)
    mark = 0                    #用于指示当前音标中游标所处的位置
    skipmark = 0                #用于记录是否进行双字符音标跳过操作，如果是1则跳过
    vector = []
    sum = 0
    #print('soundmark=',soundmark,'len=', length,' turn=',turn)

    for m in soundmark:
        if skipmark == 1:
            mark += 1
            skipmark = 0
            continue
        if mark+1 < length:
            m2 = soundmark[mark+1]
        if length >1:
            None
            # print(m , m2, mark , length)
        if m == 'i':
            for i in range(mark, mark + turn) :
                vector.append(-0.56)
                sum += 1
            mark += 1
        elif m== 'ɪ':
            for i in range(mark, mark + turn) :
                vector.append(-0.54)
                sum += 1
            mark += 1
        elif m== 'ɛ':
            for i in range(mark, mark + turn) :
                vector.append(-0.52)
                sum += 1
            mark += 1
        elif m== 'æ':
            for i in range(mark, mark + turn) :
                vector.append(-0.5)
                sum += 1
            mark += 1
        elif m== 'ɝ':
            for i in range(mark, mark + turn) :
                vector.append(-0.48)
                sum += 1
            mark += 1
        elif m== 'ə':
            for i in range(mark, mark + turn) :
                vector.append(-0.46)
                sum += 1
            mark += 1
        elif m== 'ʌ':
            for i in range(mark, mark + turn) :
                vector.append(-0.44)
                sum += 1
            mark += 1
        elif m== 'u':
            for i in range(mark, mark + turn) :
                vector.append(-0.42)
                sum += 1
            mark += 1
        elif m== 'ʊ':
            for i in range(mark, mark + turn) :
                vector.append(-0.4)
                sum += 1
            mark += 1
        elif m== 'ɔ':
            if m2 == 'ɪ':
                for i in range(mark, mark + 2*turn):
                    vector.append(-0.38)
                    sum += 1
                skipmark = 1
                mark += 1
            else:
                for i in range(mark, mark + turn) :
                    vector.append(-0.36)
                    sum += 1
                mark += 1
        elif m== 'ɑ':
            for i in range(mark, mark + turn) :
                vector.append(-0.34)
                sum += 1
            mark += 1
        elif m== 'ˈ':
            for i in range(mark, mark + turn) :
                vector.append(-0.32)
                sum += 1
            mark += 1
        elif m== 'eɪ':
            for i in range(mark, mark + turn) :
                vector.append(-0.3)
                sum += 1
            mark += 1
        elif m == 'a':
            if m2 == 'ɪ':
                for i in range(mark, mark + 2*turn):
                    vector.append(-0.28)
                    sum += 1
                skipmark = 1
                mark += 1
            elif m2 =='ʊ':
                for i in range(mark, mark + 2*turn) :
                    vector.append(-0.26)
                    sum += 1
                skipmark = 1
                mark += 1
            else:
                for i in range(mark, mark + turn) :
                    vector.append(-0.24)
                    sum += 1
                mark += 1
        elif m == 'aʊ':
            for i in range(mark, mark + turn) :
                vector.append(-0.22)
                sum += 1
            mark += 1
        elif m == 'ɪr':
            for i in range(mark, mark + turn) :
                vector.append(-0.20)
                sum += 1
            mark += 1
        elif m == 'ɛr':
            for i in range(mark, mark + turn) :
                vector.append(-0.18)
                sum += 1
            mark += 1
        elif m == 'ʊr':
            for i in range(mark, mark + turn) :
                vector.append(-0.16)
                sum += 1
            mark += 1
        elif m == 'p':
            for i in range(mark, mark + turn) :
                vector.append(-0.14)
                sum += 1
            mark += 1
        elif m =='t':
            if m2 == 'r':
                for i in range(mark, mark + 2*turn):
                    vector.append(-0.12)
                    sum += 1
                skipmark = 1
                mark += 1
            elif m2 =='s':
                for i in range(mark, mark + 2*turn) :
                    vector.append(-0.1)
                    sum += 1
                skipmark = 1
                mark += 1
            else:
                for i in range(mark, mark + turn) :
                    vector.append(-0.08)
                    sum += 1
                mark += 1
        elif m =='k':
            for i in range(mark, mark + turn) :
                vector.append(-0.06)
                sum += 1
            mark += 1
        elif m =='b':
            for i in range(mark, mark + turn) :
                vector.append(-0.04)
                sum += 1
            mark += 1
        elif m =='d':
            if m2 == 'r':
                for i in range(mark, mark + 2*turn):
                    vector.append(-0.02)
                    sum += 1
                skipmark = 1
                mark += 1
            elif m2 =='ʒ':
                for i in range(mark, mark + 2*turn) :
                    vector.append(0)
                    sum += 1
                skipmark = 1
                mark += 1
            elif m2 == 'z':
                for i in range(mark, mark + 2*turn):
                    vector.append(0.02)
                    sum += 1
                skipmark = 1
                mark += 1
            else:
                for i in range(mark, mark + turn) :
                    vector.append(0.04)
                    sum += 1
                mark += 1
        elif m =='g':
            for i in range(mark, mark + turn) :
                vector.append(0.06)
                sum += 1
            mark += 1
        elif m =='f':
            for i in range(mark, mark + turn) :
                vector.append(0.08)
                sum += 1
            mark += 1
        elif m =='s':
            for i in range(mark, mark + turn) :
                vector.append(0.1)
                sum += 1
            mark += 1
        elif m =='ʃ':
            for i in range(mark, mark + turn) :
                vector.append(0.12)
                sum += 1
            mark += 1
        elif m =='θ':
            for i in range(mark, mark + turn) :
                vector.append(0.14)
                sum += 1
            mark += 1
        elif m =='h':
            for i in range(mark, mark + turn) :
                vector.append(0.16)
                sum += 1
            mark += 1
        elif m =='v':
            for i in range(mark, mark + turn) :
                vector.append(0.18)
                sum += 1
            mark += 1
        elif m =='z':
            for i in range(mark, mark + turn) :
                vector.append(0.2)
                sum += 1
            mark += 1
        elif m =='ʒ':
            for i in range(mark, mark + turn) :
                vector.append(0.22)
                sum += 1
            mark += 1
        elif m =='ð':
            for i in range(mark, mark + turn) :
                vector.append(0.24)
                sum += 1
            mark += 1
        elif m =='r':
            for i in range(mark, mark + turn) :
                vector.append(0.26)
                sum += 1
            mark += 1
        elif m =='m':
            for i in range(mark, mark + turn) :
                vector.append(0.28)
                sum += 1
            mark += 1
        elif m =='n':
            for i in range(mark, mark + turn) :
                vector.append(0.3)
                sum += 1
            mark += 1
        elif m =='ŋ':
            for i in range(mark, mark + turn) :
                vector.append(0.32)
                sum += 1
            mark += 1
        elif m =='l':
            for i in range(mark, mark + turn) :
                vector.append(0.34)
                sum += 1
            mark += 1
        elif m =='j':
            for i in range(mark, mark + turn) :
                vector.append(0.36)
                sum += 1
            mark += 1
        elif m =='w':
            for i in range(mark, mark + turn) :
                vector.append(0.38)
                sum += 1
            mark += 1
        elif m == 'ː':
            for i in range(mark, mark + turn):
                vector.append(0.4)
                sum += 1
            mark += 1
        elif m == 'ʤ':
            for i in range(mark, mark + turn):
                vector.append(0.42)
                sum += 1
            mark += 1
        elif m == 'ɒ':
            for i in range(mark, mark + turn):
                vector.append(0.44)
                sum += 1
            mark += 1
        elif m == 'e':
            print('m=',m,'m2=',m2,'mark=',mark,'mark=',mark)
            if m2 =='ɪ':
                for i in range(mark, mark + 2*turn):
                    vector.append(0.46)
                    sum += 1
                skipmark = 1
                mark += 1
        elif m == 'ʧ':
            for i in range(mark, mark + turn):
                vector.append(0.48)
                sum += 1
            mark += 1
        elif m == 'ˌ':
            for i in range(mark, mark + turn):
                vector.append(0.5)
                sum += 1
            mark += 1
        elif m == 'ɜ':
            for i in range(mark, mark + turn):
                vector.append(0.52)
                sum += 1
            mark += 1
    if length*turn != 100:
        for i in range(length * turn, 100):
                vector.append(0.54)
                sum += 1
    #print('sum=', sum)
    return vector


def test():
    # 这个方法是用来整理好单词与音标文件的
    f = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\20000soundmark.txt', mode='r',encoding='utf-8')
    fw = open('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\test3.txt', mode='w', encoding='utf-8')
    line =f.readline()
    tline =''
    while line:
        line2 = f.readline()
        if line == line2 == '\n' or line == '\n' and line2 ==' \n':
            # fw.write('\n')
            # line = f.readline()
            print('s')
            None
        elif  line!=line2 and line2 == '\n':
            fw.write(line)
        elif  line!=line2 and line2 != '\n':
            fw.write(line)
        line = line2
    fw.close()
    f.close()

def readDict(dir):
    f = open(dir, 'r',encoding ='utf-8')    #英文单词总量20544
    dict = eval(f.read())
    print('---------------------字典加载完成------------------------')
    word = input()
    while word:
        print('soundmark = ',dict[word]['soundmark'])         #单词对应音标  type = str
        print('\ntag_weight = ',dict[word]['tag_weight'])        #词性表示35维  type = list 每一维代表一种词性
        print('\ntext = ',dict[word]['text'])              #文本表示300维 type = list
        print('\nvoice = ',dict[word]['voice'])             #声音表示100维 type = list 音标序列映射得到
        print('\nid = ',dict[word]['id'])
        word = input()
    f.close()

if __name__ == '__main__':
    # a = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy')
    # wordList = a.tolist()
    #test()
    #initSoundMark()
    readDict('C:\\Users\Administrator\Desktop\论文\数据\词性信息\\Edict3.txt')
    #print(getWordTagWeight(wordList,'dog'))
    #setTagWeight()


'''
test_word1 = word.word()
test_word1.setWord('jj', 'CC')
test_word2 = word.word()
test_word2.setWord('jj', 'CD')
test_word3 = word.word()
test_word3.setWord('jj', 'DT')
test_list = test_word1, test_word2, test_word3]
np.save('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy', test_list)
'''

'''
#下面是统计词典中单词对应词性的代码***************************************************************************************
a = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy')
wordList = a.tolist()
mark = 0
print('word list 读取完毕--------------')
#这边准备要存储的单词
with open(r"C:\\Users\Administrator\Desktop\论文\数据\词性信息\7.txt", "r+") as f:
    str = f.read()
str = nltk.word_tokenize(str)
word_tag = nltk.pos_tag(str)
print('单词列表准备完毕----------------')
#开始存储

for item in word_tag:
    #print(item[0])
    #print(item[1])
    for w in wordList:
        #print(w.word, '      ', item[0])
        if w.word == item[0]:
            w.addChara(item[1])
            mark = 0
            break
        else:
            mark = 1
    if mark == 1:
        print('没有单词', item[0], '的信息，插入信息')
        new_word = word.word()
        new_word.setWord(item[0], item[1])
        wordList.append(new_word)

wordList = np.array(wordList)
#showWordList(wordList)
np.save('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy', wordList)
print('word list 保存完毕----------------------list长度=', getWordListLength(wordList))
#initWordList(wordList)  #初始化存储列表
a = np.load('C:\\Users\Administrator\Desktop\论文\数据\词性信息\word.npy')
wordList = a.tolist()
'''









'''
gensim.models['word']
def getword():
    return 'hello world '


def getsupersense(word):
    supersenseList=wn.synsets(word)
    print(supersenseList)
    #print(dir(nltk.corpus.reader.wordnet.Synset)); 
    print(supersenseList[0].root_hypernyms())
    print(supersenseList[0].hypernyms())
    return None


getsupersense('cat')
getsupersense('rain')
'''
'''
# word2vec Text8 的训练
def train_save_model():
    # logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)',level=logging.INFO)
    # 加载预料
    #sentences = word2vec.Text8Corpus('C:\\Users\Administrator\Desktop\论文\技术预备\第一阶段\\enwik8')
    model = word2vec.Word2Vec('C:\\Users\Administrator\Desktop\论文\技术预备\第一阶段\\MyDate.text', size=200)
    model.save('C:\\Users\Administrator\Desktop\论文\技术预备\第一阶段\\text.model')
# 加载模型
def load_model():
    model = word2vec.Word2Vec.load('text.model')
    # simi = model.similar_by_vector('women', 'men')
    # print(simi)
    print(model.most_similar('man'))
    print(model['red'])
# 执行代码
#load_model()
train_save_model()
'''



