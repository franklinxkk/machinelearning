import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image,ImageDraw,ImageFont
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 处理数据
def handle_data():
	movies = pd.read_csv('movie_data.csv')
	print("处理{0}部电影数据".format(movies.shape[0]))
	df = movies[['title','director','actors','character','genre']]
	df['director'] = df.director.apply(lambda x: x.split('|'))
	df['actors'] = df.actors.apply(lambda x: x.split('|'))
	df['tag_of_movie'] = ''
	for index, row in df.iterrows():
		words = row['director']
		words.extend(row['actors'])
		words.append(row['character'])
		words.append(row['genre'])
		row['tag_of_movie'] = ' '.join(words)
	df = df[['title','tag_of_movie']]
	return df
def train_model(df):
	print("训练电影模型")
	count = CountVectorizer()
	count_matrix = count.fit_transform(df['tag_of_movie'])
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
        print("还未学习该电影的数据,试试其他电影吧")
        return recommend_movies
    idx = target.index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    # 10个距离最近的电影
    top_10_indices = list(score_series.iloc[1:11].index)
    print("生成电影({0})的推荐结果".format(title))
    for i in top_10_indices:
        movie_title = list(df['title'])[i]
        if movie_title not in recommend_movies and movie_title != title:
            recommend_movies.append(movie_title)
    print("\n".join(recommend_movies))
    draw_movie_img(title,recommend_movies)
    return recommend_movies

# 生成推荐的电影图片
def draw_movie_img(title,recommend_movies):
	fontpath = ImageFont.truetype("ziti.TTF", 30)
	s = os.getcwd()
	path1 = s + "/temp.png"
	im1 = Image.open(path1)  # 文件存在的路径
	im2 = Image.new("RGB",im1.size)   # 矩形背景颜色
	draw = ImageDraw.Draw(im1)   # 定义图片
	h = 80
	top = 280
	draw.text((200, top+h), title, fill=(255, 255, 255), font=fontpath)     # 添加文字
	h = h + 100
	for i in recommend_movies:
		draw.text((200, top+h), i, fill=(255, 255, 255), font=fontpath)
		h = h + 50
	im2.paste(im1)
	im2.save("xiguatuijian.png")
	show_plt_img(im2)

def show_plt_img(im2):
	fig = plt.figure()
	fig.canvas.set_window_title('电影推荐')
	plt.axis('off')
	plt.imshow(im2)
	plt.show()

def show_sys_img(img):
	from PIL import Image
	img.show()

# 通过二进制保存电影图片	
def save_movie_img(imgurl,imgdata):
    with open(imgurl, 'wb') as file:
        file.write(imgdata)	
