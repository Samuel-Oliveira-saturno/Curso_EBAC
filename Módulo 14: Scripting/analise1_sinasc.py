import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

sns.set_theme()

# Função para criar os gráficos
def plota_pivot_table(df, value, index, func, ylabel, xlabel, opcao='nada'):
    if opcao == 'nada':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).plot(figsize=[15, 5])
    elif opcao == 'sort':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).sort_values(value).plot(figsize=[15, 5])
    elif opcao == 'unstack':
        pd.pivot_table(df, values=value, index=index,
                       aggfunc=func).unstack().plot(figsize=[15, 5])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return None

# Função para processar os dados com base no mês
def processa_dados(mes):
    # Carregar o dataframe
    arquivo = f'SINASC_RO_2019_{mes}.csv'
    if os.path.exists(arquivo):
        sinasc = pd.read_csv(arquivo)
    else:
        print(f"O arquivo {arquivo} não foi encontrado.")
        return

    max_data = sinasc.DTNASC.max()[:7]

    # Criar uma pasta no diretório do usuário
    os.makedirs(f'./output_2/figs/{max_data}', exist_ok=True)

    # Plotar e salvar os gráficos com as variáveis relevantes
    plota_pivot_table(sinasc, 'IDADEMAE', 'DTNASC', 'count', 'Quantidade de nascimentos', 'Data de nascimento')
    plt.savefig(f'./output_2/figs/{max_data}/quantidade_de_nascimentos_{mes}.png')

    plota_pivot_table(sinasc, 'IDADEMAE', ['DTNASC', 'SEXO'], 'mean', 'Média da idade da mãe', 'Data de nascimento', 'unstack')
    plt.savefig(f'./output_2/figs/{max_data}/media_idade_mae_por_sexo_{mes}.png')

    plota_pivot_table(sinasc, 'PESO', ['DTNASC', 'SEXO'], 'mean', 'Média do peso do bebê', 'Data de nascimento', 'unstack')
    plt.savefig(f'./output_2/figs/{max_data}/media_peso_bebe_por_sexo_{mes}.png')

    plota_pivot_table(sinasc, 'PESO', 'ESCMAE', 'median', 'Média do Apgar1', 'Gestação', 'sort')
    plt.savefig(f'./output_2/figs/{max_data}/media_apgar1_por_escolaridade_mae_{mes}.png')

    plota_pivot_table(sinasc, 'APGAR1', 'GESTACAO', 'mean', 'Média do Apgar1', 'Gestação', 'sort')
    plt.savefig(f'./output_2/figs/{max_data}/media_apgar1_por_gestacao_{mes}.png')

    plota_pivot_table(sinasc, 'APGAR5', 'GESTACAO', 'mean', 'Média do Apgar5', 'Gestação', 'sort')
    plt.savefig(f'./output_2/figs/{max_data}/media_apgar5_por_gestacao_{mes}.png')

# Se o script for chamado diretamente
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Por favor, forneça o(s) mês(es) como argumento(s).")
        sys.exit(1)
    meses = sys.argv[1:]
    for mes in meses:
        mes_upper = mes.upper()
        processa_dados(mes_upper)
