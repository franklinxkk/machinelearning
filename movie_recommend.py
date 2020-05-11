import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings

# 读取数据
warnings.filterwarnings("ignore")
def read_data():
	movies = pd.read_csv('movie.csv')
	print("读取电影数据-{0}部".format(movies.shape[0]))
	return movies

# 处理数据
def handle_data(movies_df):
	df = movies_df[['title','director','actors','character','genre']]
	df['director'] = df['director'].map(lambda x: x.split('|')[:5])
	df['actors'] = df['actors'].map(lambda x: x.split('|')[:5])
	df['bag_of_words'] = ''
	columns = ['genre', 'director', 'actors', 'character']
	for index, row in df.iterrows():
		words = ''
		for col in columns:
			words += ' '.join(row[col]) + ' '
			row['bag_of_words'] = words
	df = df[['title','bag_of_words']]
	print("处理电影数据-完成")
	return df
def train_model(df):
	print("电影模型训练-完成")
	count = CountVectorizer()
	count_matrix = count.fit_transform(df['bag_of_words'])
	cosine_sim = cosine_similarity(count_matrix, count_matrix)
	indices = pd.Series(df['title'])
	return cosine_sim,indices,df

# 推荐
def recommend_movie(title, model):
    cosine_sim = model[0]
    indices = model[1]
    df = model[2]
    recommend_movies = []
    target = indices[indices == title]
    if len(target) <= 0:
        print("请输入完整的电影名称")
        return recommend_movies
    idx = target.index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    # 10个距离最近的电影
    top_10_indices = list(score_series.iloc[1:11].index)
    print("生成电影({0})的推荐结果-完成".format(title))
    for i in top_10_indices:
        movie_title = list(df['title'])[i]
        recommend_movies.append(movie_title)
    return recommend_movies
