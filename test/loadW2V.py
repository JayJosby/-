import gensim
import numpy as np
from nltk.corpus import wordnet as wn
#model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\Administrator\Desktop\论文\数据\预训练词向量模型\GoogleNews-vectors-negative300.bin',binary=True)
print("模型GoogleNews-vectors-negative300.bin加载完成")


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return 0
        else:
            return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)

def similarity1(x, y):
    score = 0
    score = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return score

# print(model['cat'].tolist(),model['dog'].tolist())
# print(round(model.similarity("dog","cat"),2))
# print(round(cos_sim(model['cat'].tolist(),model['dog'].tolist()),2))
# print(round(cosine_similarity(model['cat'].tolist(),model['dog'].tolist()),2))
# print(round(similarity1(model['cat'].tolist(),model['dog'].tolist()),2))

word = input()
while word:
    tag = [0,0,0,0,0]

    for key in wn.synsets(word):
        print(key.pos())
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
    print(tag)
    word = input()