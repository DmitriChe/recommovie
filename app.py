import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import re
import string
import requests
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import GigaChat

api_key = st.secrets["api_keys"]["gigachat"]

def clean(text):
    text = str(text)
    text = text.lower()  # нижний регистр
    text = re.sub(r"http\S+", " ", text)  # удаляем ссылки
    text = re.sub(r"@\w+", " ", text)  # удаляем упоминания пользователей
    text = re.sub(r"#\w+", " ", text)  # удаляем хэштеги
    text = re.sub(r"\d+", " ", text)  # удаляем числа
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"<.*?>", " ", text)  #
    text = re.sub(r"[️«»—]", " ", text)
    text = text.lower()
    return text

def get_embedding(text):
    out = model.encode(text)
    return out

def give_recommendations(query, top_k=10):
    """Provide movie recommendations based on the query input."""
    query_embedding = model.encode(query, convert_to_tensor=True).cpu()
    similarities = util.pytorch_cos_sim(query_embedding, film_embeddings)[0]
    top_results = similarities.cpu().numpy().argsort()[::-1][:top_k]
    top_movies = df.iloc[top_results].copy()
    similarity_scores = similarities.cpu().numpy()[top_results].copy()
    top_movies['similarity_score'] = similarity_scores
    return top_movies

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

st.title('💡AI movie Recommendator')

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Чтение датасета из csv и удаление пустых колонок 'Year' и 'Rating'
df = pd.read_csv('./source/movies.csv')
# Переименование колонок в соотвествии с заданием и удаление пустых колонок 'Year' и 'Rating'
df = df.rename(columns={
    'Title': 'movie_title',
    'Description': 'description',
    'Poster URL': 'image_url',
    'Page URL': 'page_url'
}).drop(['Year', 'Rating'], axis=1)

film_embeddings = np.load('./models/film_embedded.npy')

query = st.text_input('What kind of movie do u whant to find🕵️? (Put description for recommendations)', '''Man who fighting with corruption in Gotham city''')

button = st.button('Get recommendation')


with st.sidebar:
    with st.expander("❓About project"):
        st.subheader("**Original Dataset** 🔘 4860 elements:")
        with st.popover("Movies Dataset Preview👁️"):
            st.write(df)

        st.subheader("**Model** from SentenceTransformer:")
        st.write("🚗distilbert-base-nli-stsb-mean-tokens")

        st.subheader("**Data Source for Parsing:**")
        st.write("[🔗letterboxd.com](https://letterboxd.com/)")

        st.subheader("**Parsing time:**")
        st.write("⏱️20 hours 🌒Selenium + Beautiful💃🧴Soup")


        st.subheader("**Generative model for movie descriptions:**")
        st.write("[🌐GigaChat (Сбер)](https://giga.chat/)")



if query.strip() or (query.strip() and button):
    return_df = give_recommendations(query).reset_index()

    # Виджет для выбора количества фильмов для отображения
    max_value = return_df.shape[0]
    st.sidebar.header(f"🔍Found {max_value} movies:")
    with st.sidebar:
        with st.popover("Recommended Movies"):
            st.write(return_df)
    st.sidebar.header("⚙️Movie Display Settings")
    num_movies = st.sidebar.number_input(f'How many movies to display? (from 1 to {max_value})', min_value=1, max_value=max_value, value=2, step=1)

    # Раскомментировать для отладки без запросов в GigaChat
    # for i in range(num_movies):
    #     col1, col2 = st.columns([1, 4])
    #     with col1:
    #         st.image(return_df['image_url'][i])
    #     with col2:
    #         st.write(f"**Movie name:** {return_df['movie_title'][i]}")
    #         st.write(f"**Similarity Score:** {np.round(float(return_df['similarity_score'][i]), 2)}")
    #         st.write(f"[See page on site]({return_df['page_url'][i]})")

    for i in range(num_movies):
        container = st.container(border=True)
        with container: 
            col = st.columns(2)
            sim_score = np.round(float(return_df['similarity_score'][i]), 2)
            name = return_df['movie_title'][i]
            col[0].image(return_df['image_url'][i])
            col[1].write(f'Movie name: {name}')
            col[1].write(f'Similarity Score: {sim_score}')
            col[1].write("[See page on site](%s)" % return_df['page_url'][i])

            
            with col[1].expander(f"Get summary plot for {name}"):
                #st.write('Movie summary plot')
                chat = GigaChat(credentials=api_key, verify_ssl_certs=False)
                messages = [
                SystemMessage(
                    content="Ты самый большой знаток фильмов,который помогает поользователю узнать краткое содержание фильма по его названию и отвечает на его вопросы без лишних вопросов"
                             )
                           ]
                user_input = f"User: Напиши мне красткое содержание фильма {name}, только описание без лишних слов"
                messages.append(HumanMessage(content=user_input))
                res = chat(messages)
                messages.append(res)
                st.write(res.content)

else:
    if button:
        st.warning('Please, put your text for recommendations')