import streamlit as st
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
<h1 style="color:black;">Movie Recommendation</h1>
<h2 style="color:black;">Enter Movie Name</h2>
""",unsafe_allow_html=True)



movie_data = pd.read_csv('Preprocessed_Data.csv')
movie_data['original_title'] = movie_data['original_title'].str.lower()
movie_data.dropna(subset=['plot'],inplace=True)
movie_data.reset_index(drop = True,inplace = True)

data_recommend = movie_data.drop(columns=['movie_id', 'original_title','plot'])
data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
data_recommend = data_recommend.drop(columns=[ 'cast','genres'])

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data_recommend['combine'])

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_data['plot'])

combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
cosine_sim = cosine_similarity(combine_sparse, combine_sparse)

data = movie_data
transform = cosine_sim


selected = st.text_input("")
button_clicked = st.button("OK")


if selected:
	search_similar_movie = selected
	suggested_search = movie_data[movie_data['original_title'].str.contains(search_similar_movie)]
	if len(suggested_search)>0:
		suggested_index = suggested_search.index[0]

		title = movie_data.iloc[suggested_index,2]

		indices = pd.Series(data.index ,index = [data['original_title']])
		index = indices[title]

		sim_scores = list(enumerate(transform[index][0]))
		sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
		sim_movies = sim_scores[1:11]

		sim_movies_indices  = [i[0] for i in sim_movies]

		recommendation_data = movie_data.iloc[sim_movies_indices,:]
		if len(recommendation_data) !=0:
			st.write(f'Movies similar to {title}')
			df = recommendation_data
			df.reset_index(inplace=True)
			df.rename(columns = {'original_title' :'Similar Movies'},inplace=True)
			st.table(df['Similar Movies'])
		else:
			st.write('No similar movie found')
	else:
		st.write('Movie not in Database')
