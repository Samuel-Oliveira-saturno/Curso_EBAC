import streamlit as st


st.title('# Bem vindo!')

st.header('Uma nova jornada se inicia!')

st.subheader('Uma nova jornada se inicia!')

st.markdown("# Texto")

st.markdown("--------")


st.markdown("Markdown")



st.markdown("Markdown itálico")


st.markdown("*Markdown itálico*_")



st.markdown("Isso é um link para [google](https://google.com.br)")

st.markdown("<h1 style='text-align: center; color: red;'>Título em Vermelho</h1>",unsafe_allow_html=True)


st.markdown("Olá Mundo :sunglasses:")


st.markdown('''MARHDOWN | PEQUENO | GRANDE
               ---      |   ---   | ----
             *ITÁLICO   | 'CODE'  | **NEGRITO**                      
''')