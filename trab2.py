
# ! -------------------- IMPORTS E CONFIGS --------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="Trabalho 2", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# ! -------------------- INTRODUÇÃO --------------------
st.title("Trabalho 2 - SCC0252 (Visualização Computacional)")
st.subheader("Visualisação Analítica dos Top 1000 Filmes do IMDb")
st.markdown("""
- Luísa Balleroni Shimabucoro - 11832385
- Wictor Dalbosco Silva - 11871027
""")

# ! -------------------- CARREGAMENTO DO DATASET --------------------
st.markdown("""## **Carregamento do Dataset**""")
st.markdown("""
O Dataset escolhido foi o [IMDB Movies Dataset](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows), sendo ele composto por dados dos top 1000 filmes e programas de TV presentes no IMBd (Internet Movie Database), a maior base de dados do gênero online.

O dataset é composto por 1000 instâncias e possui 16 atributos:
- **Poster_link** - link para a imagem do poster usado pelo IMBd
- **Series_Title** - nome do filme
- **Release_Year** - ano em que foi lançado
- **Certificate** - certificado ganho pelo filme
- **Runtime** - tempo de duração do filme (em minutos)
- **Genre** - gênero do filme
- **IMDB_Rating** - nota do filme no site do IMDb
- **Overview** - resumo do filme
- **Meta_score** - nota do filme no Metacritic
- **Director** - nome do diretor
- **Star1, Star2, Star3, Star4** - nome das estrelas do filme
- **No_of_Votes** - quantidades de votos do filme
- **Gross** - quanto dinheiro o filme arrecadou
""");

# leitura do dataset a partir do csv (considerando que ele está no mesmo diretório do notebook)
dataset = pd.read_csv("./imdb_top_1000.csv")

# mostrar dataset
st.dataframe(dataset[1:].head(5))

# ! -------------------- PRÉ-PROCESSAMENTO --------------------
dataset.drop(columns=['Certificate'], inplace=True)
dataset['Meta_score'] = dataset['Meta_score'].fillna(int(dataset['Meta_score'].mean()))
# transformação da string em um valor numérico
dataset['Gross'] = dataset['Gross'].str.replace(',', '').astype(float)
dataset['Gross'] = dataset['Gross'].fillna(int(dataset['Gross'].mean()))
dataset['Runtime'] = dataset['Runtime'].str.replace(' min', '').astype(float)
# criamos dataframe com os gêneros dos filmes separados
genre_df = dataset['Genre'].str.split(",", n=-1, expand=True)
genre_df.columns = ['G1','G2','G3']  
# substituímos os valores nulos por "Undefined" já que nem todos os filmes
# possuem mais de um gênero atribuído a eles
genre_df.fillna("Undefined", inplace=True)
# retiramos a coluna onde o gênero dos filmes estão agrupados
dataset.drop(columns=['Genre'], inplace=True)
# adicionamos as novas colunas que separam os gêneros dos filmes
dataset = pd.concat([dataset, genre_df], axis=1)

# ! -------------------- ANÁLISE DE CORRELAÇÃO --------------------
st.markdown("""## **Visualização de dados**""")

# ----- Análise de correlação geral -----
st.markdown("""
### Análise de correlação
#### *Correlação geral*

A partir dos gráficos de Heatmap abaixo, podemos verificar que os atributos que possuem maior correlação são `Gross` e o `No_of_Votes` e `No_of_Votes` e `IMDB_Rating`, ou seja, existe uma relação de dependência positiva, ou seja se um cresce o outro tende a crescer, razoável entre o número de votos na plataforma e o dinheiro arrecadado com um filme e também da quantidade de votos com a nota no IMDb. Para o restante dos atributos a correlação é muito baixa ou quase nula.
""")
fig = px.imshow(dataset.corr().round(2), text_auto=True, color_continuous_scale='Viridis', width=550, height=550)
st.plotly_chart(fig, use_container_width = True)

col_left, col_right = st.columns(2)

# ----- Análise de correlação por gênero (heatmap) -----
with col_left:
    st.markdown("#### *Correlação por gênero*")
    option = st.selectbox(
    'Escolha o gênero a ser analisado:',
    ('Drama', 'Action', 'Comedy', 'Crime', 'Biography', 'Animation'))
    st.markdown("""
    Pelos gráficos é perceptível que existe uma diferença considerável em relação a correlação entre os atributos dentro de cada gênero. Por exemplo, a relação entre o número de votos e do faturamento é muito maior para filmes de Animação e Ação do que para os outros tipos. Além disso, alguns gêneros possuem comportamentos únicos, como Biografia, em que existe uma dependência negativa do faturamento com a nota no Metacritic e Comédia, que é o único gênero em que a nota no IMBd e a quantidade de votos não possuem quase nenhuma relação.
    """) 

with col_right:
    fig = px.imshow(dataset.groupby(['G1']).get_group(option).corr().round(2), text_auto=True, color_continuous_scale='Viridis', width=550, height=550)
    # fig.update_layout(title=dict(text='Animação',font=dict(size=20), x=.5, y=.92), bargap=0.1)
    st.plotly_chart(fig, use_container_width = True)


# ----- Análise de correlação por gênero (pairplot) -----
st.markdown("#### *Correlação por Pair Plots*")
st.markdown("""Podemos também fazer uma análise mais visual da correlação por meio de Pair Plots dos atributos numéricos, selecionando um ou mais gêneros para analisar a correlação dentre os atributos numéricos.""")

# gêneros com menos de 10 samples são agregados na categoria 'Other'
aux_df = dataset.copy()
aux_df['G1'].loc[~aux_df['G1'].isin(['Drama', 'Action', 'Comedy', 'Crime', 'Biography', 'Animation', 'Adventure', 'Mystery', 'Horror'])] = 'Other'

options = st.multiselect(
    'Escolha seus gêneros favoritos de filmes',
    ['Drama', 'Action', 'Comedy', 'Crime', 'Biography', 'Animation', 'Adventure', 'Mystery', 'Horror', 'Other'],
    ['Drama', 'Action', 'Comedy'])

fig = px.scatter_matrix(aux_df.loc[aux_df['G1'].isin(options)],
    dimensions=['Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross'],
    color="G1")

st.plotly_chart(fig, use_container_width = True)


# ! -------------------- ANÁLISE DE ATRIBUTOS NUMÉRICOS --------------------
st.markdown("""### Análise de atributos numéricos""")
st.markdown("""#### *Distribuição de filmes e votos ao longo dos anos*""")
st.markdown("""Pode ser feita uma análise da distribuição dos filmes mais bem avaliados no IMDb em termos do seu ano de lançamento. De maneira geral, é possível observar que filmes lançados a partir dos anos 2000 predominam nessa lista. Quando observamos o gráfico de votos em relação ao ano lançado é possível verificar que a proporção de votos para quantidade de filmes é bem proporcional para filmes lançados após os anos 2000, mas filmes mais antigos, em média, receberam menos votos.""")

vc = list(aux_df['Released_Year'].unique())[:-1]
sorted_years = sorted([int(year) for year in vc])

start_color, end_color = st.select_slider(
    'Selecione o intervalo de tempo a ser analisado:',
    options=sorted_years,
    value=(1950, 2010))

# criação da lista de datas seleciadas pelo usuário
aux_hist = dataset.copy()
date_list = list(range(int(start_color), int(end_color)+1))
date_list = [str(date) for date in date_list]

# Análise de quantidade de filmes de cada ano
aux_hist = aux_hist.loc[aux_hist['Released_Year'].isin(date_list)]
fig = px.histogram(data_frame=aux_hist.sort_values(by='Released_Year'), x='Released_Year', nbins=len(date_list))
fig.update_layout(title=dict(text='Distribuição de filmes de 1920 - atualmente',
                           font=dict(size=20), x=.1), bargap=0.1)
fig.update_traces(marker_color='Slateblue')
st.plotly_chart(fig, use_container_width = True)

# Análise de quantidade de votos dos filmes de cada ano
fig = px.bar(aux_hist.sort_values(by='Released_Year'), x='Released_Year', y='No_of_Votes')
fig.update_traces(marker_color='Slateblue')
fig.update_layout(title=dict(text='Ano de lançamento x Número de Votos recebidos',
                           font=dict(size=20), x=.5))
st.plotly_chart(fig, use_container_width = True)


st.markdown("""#### *Distribuição de dados numéricos dos filmes*""")
st.markdown("""Através dos Histogramas em conjunto com os seus respectivos Box Plots podemos analisar, simultaneamente, a distribuição do Metascore, IMDb Rating, Runtime, Número de Votos e Faturamento.""")
# histograma do Meta_score
fig = px.histogram(dataset['IMDB_Rating'], marginal='box')
fig.update_layout(title=dict(text='Distribuição das notas do IMDb',
                        font=dict(size=20), x=.5), bargap=0.1)
fig.update_traces(marker_color='Hotpink')
st.plotly_chart(fig, use_container_width = True)

col_left, col_right = st.columns(2)


with col_left:
    # histograma do Meta_score
    fig = px.histogram(dataset['Meta_score'], marginal='box')
    fig.update_layout(title=dict(text='Distribuição de Metascore',
                           font=dict(size=20), x=.5), bargap=0.1)
    fig.update_traces(marker_color='Hotpink')
    st.plotly_chart(fig, use_container_width = True)

    # histograma do número de votos
    fig = px.histogram(dataset['No_of_Votes'], marginal='box')
    fig.update_layout(title=dict(text='Distribuição do No de Votos',
                           font=dict(size=20), x=.5), bargap=0.1)
    fig.update_traces(marker_color='Hotpink')
    st.plotly_chart(fig, use_container_width = True)

with col_right:
    # histograma do runtime
    fig = px.histogram(dataset['Runtime'], marginal='box')
    fig.update_layout(title=dict(text='Distribuição do Runtime',
                           font=dict(size=20), x=.5), bargap=0.1)
    fig.update_traces(marker_color='Hotpink')
    st.plotly_chart(fig, use_container_width = True)  

    # histograma do faturamento
    fig = px.histogram(dataset['Gross'], marginal='box')
    fig.update_layout(title=dict(text='Distribuição do Faturamento',
                           font=dict(size=20), x=.5), bargap=0.1)
    fig.update_traces(marker_color='Hotpink')
    st.plotly_chart(fig, use_container_width = True)    

# ! -------------------- ANÁLISE DE ATRIBUTOS CATEGÓRICOS --------------------
st.markdown("""### Análise de variáveis categóricas""")
st.markdown("""#### *Pesquisa por diretor*""")
st.markdown("""Podemos realizar uma busca dentro da base de dados por obras de diretores específicos, sendo que ela retorna o cartaz, título, ano de lançamento, gênero, notas e duração de cada filme do diretor escolhido.""")

# Exploração de filmes por diretor na base de dados
director = st.text_input('Digite o nome do diretor', 'Hayao Miyazaki')
d = dataset.loc[dataset['Director'] == director]

if d.empty:
    st.write("O diretor não consta na base de dados.")
else:
    with st.expander(f"Ver {d.shape[0]} filme(s) dirigidos por {director}"):
        for movie_idx in range(d.shape[0]):
            st.image(d.iloc[movie_idx]['Poster_Link'])
            st.markdown(f"""
            **Título:** {d.iloc[movie_idx]['Series_Title']}\n
            **Ano de lançamento:** {d.iloc[movie_idx]['Released_Year']}\n
            **Gênero:** {d.iloc[movie_idx]['G1']}\n
            **Nota no IMDb:** {d.iloc[movie_idx]['IMDB_Rating']}\n
            **Nota no Metacritic:** {d.iloc[movie_idx]['Meta_score']}\n
            **Duração:** {d.iloc[movie_idx]['Runtime']}min\n\n
            """)


st.markdown("""#### *Análise de contribuições de diretores*""")
st.markdown("""Aqui verificamos os diretores possuem mais obras na lista de Top 1000 e quais tiveram o maior faturamento cumulativo.""")
# gráfico de barras dos 10 diretores que mais aparecem nos dados
fig = px.bar(x=dataset['Director'].value_counts().keys()[:10], y=dataset['Director'].value_counts().values[:10], orientation='v', color=dataset['Director'].value_counts().keys()[:10], color_discrete_sequence=px.colors.qualitative.Plotly)
fig.update_layout(title=dict(text='Diretores com mais filmes no Top 1000 IMDb', font=dict(size=20), x=.5), bargap=0.1)
st.plotly_chart(fig, use_container_width = True)

# gráfico de barras dos 10 diretores com maior faturamento cumulativo
fig = px.bar(x=dataset.groupby('Director').sum().sort_values(by='Gross', ascending=False).index[:10], y=dataset.groupby('Director').sum().sort_values(by='Gross', ascending=False)['Gross'][:10], color=dataset.groupby('Director').sum().sort_values(by='Gross', ascending=False).index[:10], color_discrete_sequence=px.colors.qualitative.Plotly)
fig.update_layout(title=dict(text='Diretores com maior faturamento cumulativo', font=dict(size=20), x=.5), bargap=0.1)
st.plotly_chart(fig, use_container_width = True)


st.markdown("""#### *Análise de contribuições de atores*""")
st.markdown("""Além da contribuição dos diretores, podemos ver também quais atores possuem mais aparições gerais no Top 1000 e também como estrelas principais.""")

# gráfico de barras dos 30 atores que mais aparecem na lista
df = pd.DataFrame(dataset)
df = pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4']])
fig = px.bar(y=df.value_counts().keys()[:30], x=df.value_counts().values[:30], color=df.value_counts().values[:30])
fig.update_layout(yaxis=dict(autorange="reversed"), title=dict(text='Top 30 atores com mais aparições no Top 1000', font=dict(size=20), x=.5), bargap=0.1)
st.plotly_chart(fig, use_container_width = True)

# gráfico de barras dos 10 atores que mais aparecem na lista como protagonistas
fig = px.bar(y=dataset['Star1'].value_counts().keys()[:15], x=dataset['Star1'].value_counts().values[:15],  color=df.value_counts().values[:15])
fig.update_layout(yaxis=dict(autorange="reversed"), title=dict(text='Top 10 atores com mais aparições como estrelas principais no Top 1000', font=dict(size=20), x=.5), bargap=0.1)
st.plotly_chart(fig, use_container_width = True)


st.markdown("""### Análise por Gênero""")
st.markdown("""Primeiramente podemos ver quais gêneros estão mais presentes dentro do Top 1000 por meio de um Pie Chart.""")

# Pie Chart dos gêneros presentes
df = dataset['G1'].value_counts()
fig = px.pie(df, values='G1', names=df.index.tolist(), title='Population of European continent')
fig.update_layout(title=dict(text='Proporção percentual de gêneros', font=dict(size=20), x=.5), bargap=0.1)
fig.update_traces(textinfo='percent+label')
st.plotly_chart(fig, use_container_width = True)

col_left, col_right = st.columns([1,2])

# Análise de atributos numéricos dos cinco principais gêneros por box plots
with col_left:
    st.markdown("#### *Distribuição de atributos numéricos por gênero*")
    st.markdown("Através dos Box Plots podemos analisar como os 5 gêneros principais se comportam em relação a direferentes atributos numéricos.")
    attr_option = st.selectbox(
    'Escolha o atributo a ser analisado:',
    ('Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross'))
    st.markdown("""
    Pelos gráficos é perceptível que existe uma diferença considerável em relação a correlação entre os atributos dentro de cada gênero. Por exemplo, a relação entre o número de votos e do faturamento é muito maior para filmes de Animação e Ação do que para os outros tipos. Além disso, alguns gêneros possuem comportamentos únicos, como Biografia, em que existe uma dependência negativa do faturamento com a nota no Metacritic e Comédia, que é o único gênero em que a nota no IMBd e a quantidade de votos não possuem quase nenhuma relação.
    """) 

df = dataset.loc[dataset['G1'].isin(['Drama', 'Action', 'Comedy', 'Crime', 'Biography'])]

with col_right:
    fig = px.box(df, x="G1", y=attr_option, hover_data=['Series_Title', 'Released_Year'])
    fig.update_traces(marker_color='Hotpink')
    st.plotly_chart(fig, use_container_width = True)

# Processamento de stopwords
stop = set(STOPWORDS)
stop.update(['us', 'one', 'will', 'said', 'now', 'well', 'man', 'may',
    'little', 'say', 'must', 'way', 'long', 'yet', 'mean',
    'put', 'seem', 'asked', 'made', 'half', 'much',
    'certainly', 'might', 'came'])
test = pd.DataFrame(dataset)[['Overview', 'G1']]
test['Overview'] = test['Overview'].apply(lambda x: [palavra for palavra in x.split() if palavra not in stop])
test['Overview'] = test['Overview'].apply(lambda x : ' '.join(x))


# WordClouds dos 5 principais gêneros
col_left, col_right = st.columns([1,2])

with col_left:
    st.markdown("""#### *Análise de descrições de cada gênero*""")
    st.markdown("Podemos, por meio de WordClouds, verificar quais são as palavras mais frequentes presentes nas descrições dadas pelo IMDb.")
    genre_option = st.selectbox(
    'Escolha o gênero a ser analisado:',
    ('Drama', 'Action', 'Comedy', 'Crime', 'Biography'))
    generate = st.button('Gerar WordCloud')
    st.markdown("""
    Com base nos resultados, podemos ver que existem algumas palavras mais frequêntes em cada gênero.""") 

with col_right:
    if generate:
        with st.spinner('Carregando...'):
            st.markdown("""#### *Biografia*""")
            fig = WordCloud(max_words=50, min_font_size=20, stopwords=stop, height=1500,width=3000,background_color="white").generate(' '.join(test.loc[test['G1'] == genre_option]['Overview']))
            #fig.update_layout(title=dict(text='Drama',font=dict(size=20), x=.5, y=.9))
            plt.axis("off")
            plt.imshow(fig)
            st.pyplot()