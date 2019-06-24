import numpy as np
import nltk
class word:
    #这个方法是独立于LPGate之外的数据预处理方法，通过这个方法，可以获得数据中，每一个单词的词性分布
    def __init__(self):
        self.word = ''
        self.tag = ''
        self.text_representation = ''
        self.voice_representation = ''
        self.sum = 0.0
        self.CC = 0.0
        self.CD = 0.0
        self.DT = 0.0
        self.EX = 0.0
        self.FW = 0.0
        self.IN = 0.0
        self.JJ = 0.0
        self.JJR = 0.0
        self.JJS = 0.0
        self.LS = 0.0
        self.KMD = 0.0
        self.NN = 0.0
        self.NNS = 0.0
        self.NNP = 0.0
        self.NNPS = 0.0
        self.PDT = 0.0
        self.PRP = 0.0
        self.PRPS = 0.0
        self.RP = 0.0
        self.SYM = 0.0
        self.VB = 0.0
        self.VBD = 0.0
        self.VBG = 0.0
        self.VBN = 0.0
        self.VBP = 0.0
        self.VBZ = 0.0
        self.WDT = 0.0
        self.WP = 0.0
        self.WPS = 0.0
        self.WRB = 0.0
        self.TO = 0.0
        self.UH = 0.0
        self.RB = 0.0
        self.RBR = 0.0
        self.RBS = 0.0

        self.freCC = 0.0
        self.freCD = 0.0
        self.freDT = 0.0
        self.freEX = 0.0
        self.freFW = 0.0
        self.freIN = 0.0
        self.freJJ = 0.0
        self.freJJR = 0.0
        self.freJJS = 0.0
        self.freLS = 0.0
        self.freKMD = 0.0
        self.freNN = 0.0
        self.freNNS = 0.0
        self.freNNP = 0.0
        self.freNNPS = 0.0
        self.frePDT = 0.0
        self.frePRP = 0.0
        self.frePRPS = 0.0
        self.freRP = 0.0
        self.freSYM = 0.0
        self.freVB = 0.0
        self.freVBD = 0.0
        self.freVBG = 0.0
        self.freVBN = 0.0
        self.freVBP = 0.0
        self.freVBZ = 0.0
        self.freWDT = 0.0
        self.freWP = 0.0
        self.freWPS = 0.0
        self.freWRB = 0.0
        self.freTO = 0.0
        self.freUH = 0.0
        self.freRB = 0.0
        self.freRBR = 0.0
        self.freRBS = 0.0
        #注意，PRP$和WP$都分别写成RPRS WPS
    def setWord(self, word , tag):
        self.word = word

        if tag == 'CC':
            self.CC = 1
        elif tag == 'CD':
            self.CD = 1
        elif tag == 'DT':
            self.DT = 1
        elif tag == 'EX':
            self.EX = 1
        elif tag == 'FW':
            self.FW = 1
        elif tag == 'IN':
            self.IN = 1
        elif tag == 'JJ':
            self.JJ = 1
        elif tag == 'JJR':
            self.JJR = 1
        elif tag == 'JJS':
            self.JJS = 1
        elif tag == 'LS':
            self.LS = 1
        elif tag == 'KMD':
            self.KMD = 1
        elif tag == 'NN':
            self.NN = 1
        elif tag == 'NNS':
            self.NNS = 1
        elif tag == 'NNP':
            self.NNP = 1
        elif tag == 'NNPS':
            self.NNPS = 1
        elif tag == 'PDT':
            self.PDT = 1
        elif tag == 'PRP':
            self.PRP = 1
        elif tag == 'RP':
            self.RP = 1
        elif tag == 'SYM':
            self.SYM = 1
        elif tag == 'VB':
            self.VB = 1
        elif tag == 'VBD':
            self.VBD = 1
        elif tag == 'VBG':
            self.VBG = 1
        elif tag == 'VBN':
            self.VBN = 1
        elif tag=='VBP':
            self.VBP = 1
        elif tag=='VBZ':
            self.VBZ = 1
        elif tag=='WDT':
            self.WDT = 1
        elif tag=='WP':
            self.WP = 1
        elif tag=='WP$':
            self.WPS = 1
        elif tag=='WRB':
            self.WRB = 1
        elif tag == 'TO':
            self.TO = 1
        elif tag == 'UH':
            self.UH = 1
        elif tag == 'PRP$':
            self.PRPS = 1
        elif tag == 'RB':
            self.RB = 1
        elif tag == 'RBR':
            self.RBR = 1
        elif tag == 'RBS':
            self.RBS = 1
        self.sum = 1
    def addChara(self,tag):
        if tag == 'CC':
            self.CC += 1
        elif tag == 'CD':
            self.CD += 1
        elif tag == 'DT':
            self.DT += 1
        elif tag == 'EX':
            self.EX += 1
        elif tag == 'FW':
            self.FW += 1
        elif tag == 'IN':
            self.IN += 1
        elif tag == 'JJ':
            self.JJ += 1
        elif tag == 'JJR':
            self.JJR += 1
        elif tag == 'JJS':
            self.JJS += 1
        elif tag == 'LS':
            self.LS += 1
        elif tag == 'KMD':
            self.KMD += 1
        elif tag == 'NN':
            self.NN += 1
        elif tag == 'NNS':
            self.NNS += 1
        elif tag == 'NNP':
            self.NNP += 1
        elif tag == 'NNPS':
            self.NNPS += 1
        elif tag == 'PDT':
            self.PDT += 1
        elif tag == 'PRP':
            self.PRP += 1
        elif tag == 'RP':
            self.RP += 1
        elif tag == 'SYM':
            self.SYM += 1
        elif tag == 'VB':
            self.VB += 1
        elif tag == 'VBD':
            self.VBD += 1
        elif tag == 'VBG':
            self.VBG += 1
        elif tag == 'VBN':
            self.VBN += 1
        elif tag == 'VBP':
            self.VBP += 1
        elif tag == 'VBZ':
            self.VBZ += 1
        elif tag == 'WDT':
            self.WDT += 1
        elif tag == 'WP':
            self.WP += 1
        elif tag == 'WP$':
            self.WPS += 1
        elif tag == 'WRB':
            self.WRB += 1
        elif tag == 'TO':
            self.TO += 1
        elif tag == 'UH':
            self.UH += 1
        elif tag == 'PRP$':
            self.PRPS += 1
        elif tag == 'RB':
            self.RB += 1
        elif tag == 'RBR':
            self.RBR += 1
        elif tag == 'RBS':
            self.RBS += 1
        self.num2ratio()

    def num2ratio(self):
        new_sum = self.sum+1
        self.freCC = self.CC / new_sum
        self.freCD = self.CD / new_sum
        self.freDT = self.DT / new_sum
        self.freEX = self.EX / new_sum
        self.freFW = self.FW / new_sum
        self.freIN = self.IN / new_sum
        self.freJJ = self.JJ / new_sum
        self.freJJR = self.JJR / new_sum
        self.freJJS = self.JJS / new_sum
        self.freLS = self.LS / new_sum
        self.freKMD = self.KMD / new_sum
        self.freNN = self.NN / new_sum
        self.freNNS = self.NNS / new_sum
        self.freNNP = self.NNP / new_sum
        self.freNNPS = self.NNPS / new_sum
        self.frePDT = self.PDT / new_sum
        self.frePRP = self.PRP / new_sum
        self.frePRPS = self.PRPS / new_sum
        self.freRP = self.RP / new_sum
        self.freSYM = self.SYM / new_sum
        self.freVB = self.VB / new_sum
        self.freVBD = self.VBD / new_sum
        self.freVBG = self.VBG / new_sum
        self.freVBN = self.VBN / new_sum
        self.freVBP = self.VBP / new_sum
        self.freVBZ = self.VBZ / new_sum
        self.freWDT = self.WDT / new_sum
        self.freWP = self.WP / new_sum
        self.freWPS = self.WPS / new_sum
        self.freWRB = self.WRB / new_sum
        self.freTO = self.TO / new_sum
        self.freUH = self.UH / new_sum
        self.freRB = self.RB / new_sum
        self.freRBR = self.RBR / new_sum
        self.freRBS = self.RBS / new_sum
        self.sum = new_sum


    def getWordChara(self, word, wordList):
        for w in wordList:
            if word == w.word:
                charaList=[self.freCC]
                charaList.append(self.freCC ) 
                charaList.append(self.freCD ) 
                charaList.append(self.freDT ) 
                charaList.append(self.freEX ) 
                charaList.append(self.freFW ) 
                charaList.append(self.freIN ) 
                charaList.append(self.freJJ ) 
                charaList.append(self.freJJR ) 
                charaList.append(self.freJJS ) 
                charaList.append(self.freLS ) 
                charaList.append(self.freKMD ) 
                charaList.append(self.freNN ) 
                charaList.append(self.freNNS )
                charaList.append(self.freNNP ) 
                charaList.append(self.freNNPS ) 
                charaList.append(self.frePDT ) 
                charaList.append(self.frePRP ) 
                charaList.append(self.frePRPS ) 
                charaList.append(self.freRP ) 
                charaList.append(self.freSYM ) 
                charaList.append(self.freVB ) 
                charaList.append(self.freVBD )
                charaList.append(self.freVBG ) 
                charaList.append(self.freVBN ) 
                charaList.append(self.freVBP ) 
                charaList.append(self.freVBZ ) 
                charaList.append(self.freWDT ) 
                charaList.append(self.freWP )
                charaList.append(self.freWPS ) 
                charaList.append(self.freWRB ) 
                charaList.append(self.freTO ) 
                charaList.append(self.freUH ) 
                charaList.append(self.freRB ) 
                charaList.append(self.freRBR ) 
                charaList.append(self.freRBS ) 
        return  charaList

str='her dog'
str = nltk.word_tokenize(str)
word_tag = nltk.pos_tag(str)
for [w,t] in word_tag:
    print(w,'  ',t)

'''for word in list:
    for sword in slist:
        if(sword['word']==word['word']):
            sword.setChara(word.tag)
            break   
        else:
            mark=1        
    if mark == 1:
        new_word=word(word.word,word.tag)#直接写一个词信息类，里面有所有的词性词频统计，和处理方法
        slist.append(new_word)
        mark=0


m=np.array(slist)
np.save('demo.npy',m)'''

