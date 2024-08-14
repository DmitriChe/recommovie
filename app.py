import streamlit as st
import pandas as pd

# Чтение датасета из csv и удаление пустых колонок 'Year' и 'Rating'
df = pd.read_csv('./source/movies.csv')
# Переименование колонок в соотвествии с заданием и удаление пустых колонок 'Year' и 'Rating'
df = df.rename(columns={
    'Title': 'movie_title',
    'Description': 'description',
    'Poster URL': 'image_url',
    'Page URL': 'page_url'
}).drop(['Year', 'Rating'], axis=1)


with st.sidebar:
    with st.popover("Movies Dataset Preview"):
        st.write(df)

st.sidebar.header("Настройки отображения фильмов")
# Виджет для выбора количества фильмов для отображения
num_movies = st.sidebar.number_input('Количество фильмов для отображения', min_value=1, max_value=50, value=5, step=1)

df = df.iloc[:num_movies,:]


st.header("УмнAIя система поиска фильмов")
user_description = st.text_input('Введите описание для фильма, который хотите посмотреть')

col1, col2 = st.columns([7, 1])

with col1:
    pass
with col2:
    button = st.button('Найти')


if user_description.strip() and button:
    st.write("---")


    for i in range(df.shape[0]):

        image_url = df.image_url[i]
        page_url = df.page_url[i]
        movie_title = df.movie_title[i]
        description = df.description[i]


        # Создаем колонки для изображения и текста
        col1, col2 = st.columns([1, 5])

        with col1:
            # Используем HTML, чтобы сделать изображение кликабельным
            st.markdown(
                f'<a href="{page_url}" target="_blank">'
                f'<img src="{image_url}" width="150"></a>',
                unsafe_allow_html=True
            )

        with col2:
            # Отображаем название и описание фильма
            # st.write(f"### {movie_title}")
            # st.write(f"### {movie_title} ({i + 1} из {df.shape[0]})")
            st.markdown(
                f"""
                <h3 style='display: inline;'>{movie_title}</h3>
                <span style='font-size: 14px; color: gray;'> ({i + 1} из {df.shape[0]})</span>
                """,
                unsafe_allow_html=True
            )
            st.write(description)

        # Разделитель между фильмами
        st.write("---")

else:
    if button:
        st.warning('Пожалуйста, введите запрос в поле описания.')