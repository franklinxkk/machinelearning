import pandas as pd
import jieba
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#加载数据
data = pd.read_csv('food_comment_data.csv')

#标记数据   4 5星表示正向情感
def make_label(star):
    if star > 3:
        return 1
    else:
        return 0
    
data['sentiment'] = data.star.apply(make_label)

#分词
jieba.load_userdict("my_dic.txt")  #分词不当对自然语言处理的影响！
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))
data['cut_comment'] = data.comment.apply(chinese_word_cut)

#生成词向量
vect = CountVectorizer()
x_train_vect = vect.fit_transform(data['cut_comment'])

#训练模型
nb = MultinomialNB()
nb.fit(x_train_vect, data['sentiment'])#(x_train_vect, y_train)

#读取预测数据集
data2test = pd.read_csv('data2test.csv')
#给预测数据分词
data2test['cut_comment'] = data2test.comment.apply(chinese_word_cut)
#生成预测数据的词向量
x_vec = vect.transform(data2test['cut_comment'])

#带入训练好的模型，查看结果
nb_result = nb.predict(x_vec)
data2test['nb_result'] = nb_result
print(data2test)