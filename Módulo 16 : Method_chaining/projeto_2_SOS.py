import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de renda",
     page_icon="https://i.pinimg.com/736x/6b/6a/f0/6b6af066aa6bb514f9143e3021d68afe.jpg",
     layout="centered",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')



#plots
fig, ax = plt.subplots(8,1,figsize=(10,70))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,50))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
sns.despine()
st.pyplot(plt)

# Boxplot por categoria:
# O boxplot é útil para visualizar a distribuição da renda por categoria. 
#Ele mostra a mediana, quartis e possíveis outliers.

st.write('## Distribuição de Renda por categoria')
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x='tipo_renda', y='renda', data=renda, ax=ax)
ax.set_title('Boxplot da Renda por Tipo de Renda')
ax.set_xlabel('Tipo de Renda')
ax.set_ylabel('Renda')
plt.xticks(rotation=45)
sns.despine()
st.pyplot(plt)


# Scatterplot com regressão:
# Um scatterplot pode ser usado para ver a relação entre duas variáveis, 
# como renda e idade, com uma linha de regressão para destacar padrões.

st.write('')
fig, ax = plt.subplots(figsize=(10, 8))
sns.regplot(x='idade', y='renda', data=renda, ax=ax)
ax.set_title('Relação entre Idade e Renda com Regressão')
ax.set_xlabel('Idade')
ax.set_ylabel('Renda')
sns.despine()
st.pyplot(plt)


# Gráfico de dispersão por grupo:
# Para visualizar a dispersão da renda ao longo do tempo para diferentes grupos, 
# podemos usar um scatterplot com cores representando grupos.

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='data_ref', y='renda', hue='estado_civil', data=renda, ax=ax)
ax.set_title('Dispersão da Renda ao Longo do Tempo por Estado Civil')
ax.set_xlabel('Data de Referência')
ax.set_ylabel('Renda')
plt.xticks(rotation=45)
sns.despine()
st.pyplot(plt)








