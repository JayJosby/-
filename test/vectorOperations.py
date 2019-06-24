import numpy as np
import tensorflow as  tf

def cos_sim(tensor_a, tensor_b):
    #求模
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor_a), axis=0))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor_b), axis=0))
    print(a_norm.eval())
    #内积
    a_b = tf.reduce_sum(tf.multiply(a_norm, b_norm))
    cosin = tf.divide(a_b, tf.multiply(a_norm, b_norm))
    # print('计算相似度', tensor_a.eval(), ' -----', tensor_b.eval(), '相似度计算完成', cosin.eval())
    return cosin
''' 
def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
'''


'''
with tf.Session() as sess:
    #求模
    x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=2))
    x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=2))
    #内积
    x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=2)
    cosin = x3_x4 / (x3_norm * x4_norm)
    cosin1 = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
    a, b, c, d, e = sess.run([x3_norm, x4_norm, x3_x4, cosin, cosin1])
    print a, b, c, d, e
'''
