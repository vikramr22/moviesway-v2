import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle

from cachecontrol import CacheControl
from cachecontrol.caches import FileCache

print(23)
st.set_page_config(layout="wide")

# ''' Backend '''

api_key = os.getenv("apikey")


@st.cache_data
def load_movies():
    movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
    loaded_movies = pd.DataFrame(movies_dict)

    loaded_movies.drop_duplicates(inplace=True)
    return loaded_movies


movies = load_movies()


@st.cache_data
def load_votes():
    vote_info = pickle.load(open("vote_info.pkl", "rb"))
    loaded_votes = pd.DataFrame(vote_info)
    return loaded_votes


vote = load_votes()

# csr_data = pickle.load(open("csr_data_tf.pkl","rb"))


@st.cache_resource
def load_model():
    loaded_model = pickle.load(open("model.pkl", "rb"))
    return loaded_model


model = load_model()


@st.cache_data
def load_csr_data():
    with open("csr_data_tf.pkl", "rb") as file:
        csr_data = pickle.load(file)
        return csr_data


csr_data = load_csr_data()


def recommend(movie_name):
    n_movies_to_recommend = 5
    idx = movies[movies["title"] == movie_name].index[0]

    distances, indices = model.kneighbors(
        csr_data[idx], n_neighbors=n_movies_to_recommend + 1
    )
    idx = list(indices.squeeze())
    df = np.take(movies, idx, axis=0)

    movies_list = list(df.title[1:])
    print(movies_list)

    recommend_movies_names = []
    recommend_posters = []
    movie_ids = []
    for i in movies_list:
        temp_movie_id = (movies[movies.title == i].movie_id).values[0]
        movie_ids.append(temp_movie_id)
        print(temp_movie_id)
        # fetch poster
        recommend_posters.append(fetch_poster(temp_movie_id))
        recommend_movies_names.append(i)
    return recommend_movies_names, recommend_posters, movie_ids


def fetch_poster(movie_id):
    # gzip for compressing request and CacheControl for caching reusable requests
    headers = {"Accept-Encoding": "gzip"}
    session = CacheControl(requests.Session(), cache=FileCache(".web_cache"))

    response = session.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}",
        headers=headers,
    )
    data = response.json()
    print(data)
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]


# ''' Frontend '''

v = st.write(
    """ <h2> <b style="color:red"> MoviesWay</b> </h2>""", unsafe_allow_html=True
)
# st.header(v)
# st.header(" :red[MoviesWay]")
st.write("###")

st.write(
    """ <p> Hii, welcome to <b style="color:red">Moviesway</b> this free movie recommendation engine suggests films based on your interest </p>""",
    unsafe_allow_html=True,
)
st.write("##")
my_expander = st.expander("Tap to Select a Movie  🌐️")
selected_movie_name = my_expander.selectbox("", movies["title"].values[:-3])

if my_expander.button("Recommend"):
    st.text("Here are few Recommendations..")
    st.write("#")
    names, posters, movie_ids = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i in range(0, 5):
        with cols[i]:
            st.write(
                f' <b style="color:#E50914"> {names[i]} </b>', unsafe_allow_html=True
            )
            # st.write("#")
            st.image(posters[i])
            id = movie_ids[i]
            st.write("________")
            vote_avg, vote_count = (
                vote[vote["id"] == id].vote_average,
                vote[vote["id"] == id].vote_count,
            )
            st.write(
                f'<b style="color:#DB4437">Rating</b>:<b> {list(vote_avg.values)[0]}</b>',
                unsafe_allow_html=True,
            )
            st.write(
                f'<b style="color:#DB4437">   Votes  </b>: <b> {list(vote_count.values)[0]} <b> ',
                unsafe_allow_html=True,
            )


# future release
# with st.sidebar:
#     st.write("Moviesway🍿")

st.write("##")
tab1, tab2 = st.tabs(["About", "Working"])

with tab1:
    st.caption("This a Content Based Movie Recommendation System")
    st.caption("In upcoming versions new movies would be added 😎")  #:blue[:sunglasses:]
with tab2:
    st.caption("It Contains Movie's from The Movie Data Base (TMDB)")
    st.caption("For more infos and ⭐ at https://github.com/vikramr22/moviesway-v2 ")
