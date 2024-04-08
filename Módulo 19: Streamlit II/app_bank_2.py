import timeit
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from PIL import Image
from io import BytesIO
import xlsxwriter


sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})


# FUNÇÃO PARA CARREGAR OS DADOS
@st.cache_data(show_spinner=True)
def load_data(file_data: str, sep: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath_or_buffer=file_data, sep=sep)
    except:
        return pd.read_excel(io=file_data)


@st.cache_data
def multiselect_filter(
    data: pd.DataFrame, col: str, selected: list[str]
) -> pd.DataFrame:
    if "all" in selected:
        return data
    else:
        return data[data[col].isin(selected)].reset_index(drop=True)


@st.cache_data
def df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


@st.cache_data
def df_to_excel(df: pd.DataFrame):
    output = BytesIO()
    writer = pd.ExcelWriter(path=output, engine="xlsxwriter")
    df.to_excel(excel_writer=writer, index=False, sheet_name="Sheet1")
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def main():
    st.set_page_config(
        page_title="EBAC | Módulo 19 | Streamlit II | Exercício 2",
        page_icon="/home/sos/Documentos/EBAC_Data_ Scientist/Módulo 19: Streamlit II/Material_de_apoio_M19_Cientista de Dados/img/telmarketing_icon.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # TÍTULO
    st.markdown(
        """
    <div style="text-align:center">
        <img src="https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/main/media/logo/newebac_logo_black_half.png" alt="ebac-logo">

---

# **Módulo 19** | Streamlit II
Caderno de **exercício 02**<br>


## Discente: Samuel Saturno
    

   

    
    """,
        unsafe_allow_html=True,
    )

    st.write("# Telemarketing analysis")
    st.markdown(body="---")

    # SIDEBAR
    # image = Image.open(fp='Módulo_19_-_Streamlit_II/Exercício_1/img/Bank-Branding.jpg')
    # st.sidebar.image(image=image)
    st.sidebar.markdown(
        body='<img src="https://storage.googleapis.com/kagglesdsdata/datasets/4759145/8066592/bank.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240408%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240408T184147Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=734c39b4298a76b19c9eaab2be420fef83815d4b13010b180b35280dbbcd13129cf03014b57e82f0f706caa119340ccebacfeff01d3453e7070314fd9cba617531145530a78aa3f9b3d210058e2447d96e21753470931888e71c4385986d84660911f93103e993789b05cdd71f2826c19668f86b8d93c59aa96808ba96950e4b1f68c8d0bcf8e7f4a3b554826cb37b802d933fab5dc2f17e9103b3ed08f4f9ecd0b4895726700b3eb5a7737320db2f27db467173f372157972f5a4fbfcf4a456a54d2f8d314871ebd574ed30813b50482cd1b06b5a98019b177cb7ca18dcd3a0060f799774b8813a2c8b8483352aa14e1122783c667ce030cbfe0bf1d1a9563f" width=100%>',
        unsafe_allow_html=True,
    
    )

    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader(
        label="Bank marketing data", type=["csv", "xlsx"]
    )

    if data_file_1 is not None:
        start = timeit.default_timer()

        bank_raw = load_data(
            # file_data="/home/sos/Documentos/EBAC_Data_ Scientist/Módulo 19: Streamlit II/bank-additional-full.csv",
            file_data=data_file_1,
            sep=";",
        )
        bank = bank_raw.copy()

        st.write("Time:", timeit.default_timer() - start)

        st.write("## Antes dos filtros")
        st.write(bank_raw)
        st.write("Quantidade de linhas:", bank_raw.shape[0])
        st.write("Quantidade de colunas:", bank_raw.shape[1])

        with st.sidebar.form(key="my_form"):
            # TIPO DE GRÁFICO
            graph_type = st.radio("Tipo de gráfico:", ("Barras", "Pizza"))

            # IDADE
            min_age = min(bank["age"])
            max_age = max(bank["age"])

            idades = st.slider(
                label="Idade:",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age),
                step=1,
            )

            # PROFISSÕES
            jobs_list = bank["job"].unique().tolist()
            jobs_list.append("all")
            jobs_selected = st.multiselect(
                label="Profissões:", options=jobs_list, default=["all"]
            )

            # ESTADO CIVIL
            marital_list = bank["marital"].unique().tolist()
            marital_list.append("all")
            marital_selected = st.multiselect("Estado Civil:", marital_list, ["all"])

            # DEFAULT
            default_list = bank["default"].unique().tolist()
            default_list.append("all")
            default_selected = st.multiselect("Default:", default_list, ["all"])

            # FINANCIAMENTO
            housing_list = bank["housing"].unique().tolist()
            housing_list.append("all")
            housing_selected = st.multiselect(
                "Tem financiamento imobiliário?", housing_list, ["all"]
            )

            # EMPRÉSTIMO
            loan_list = bank["loan"].unique().tolist()
            loan_list.append("all")
            loan_selected = st.multiselect("Tem empréstimo?", loan_list, ["all"])

            # CONTATO
            contact_list = bank["contact"].unique().tolist()
            contact_list.append("all")
            contact_selected = st.multiselect("Meio de contato:", contact_list, ["all"])

            # MÊS DO CONTATO
            month_list = bank["month"].unique().tolist()
            month_list.append("all")
            month_selected = st.multiselect("Mês do contato:", month_list, ["all"])

            # DIA DA SEMANA
            day_of_week_list = bank["day_of_week"].unique().tolist()
            day_of_week_list.append("all")
            day_of_week_selected = st.multiselect(
                "Dia da semana do contato:", day_of_week_list, ["all"]
            )

            bank = (
                bank.query("age >= @idades[0] and age <= @idades[1]")
                .pipe(multiselect_filter, "job", jobs_selected)
                .pipe(multiselect_filter, "marital", marital_selected)
                .pipe(multiselect_filter, "default", default_selected)
                .pipe(multiselect_filter, "housing", housing_selected)
                .pipe(multiselect_filter, "loan", loan_selected)
                .pipe(multiselect_filter, "contact", contact_selected)
                .pipe(multiselect_filter, "month", month_selected)
                .pipe(multiselect_filter, "day_of_week", day_of_week_selected)
            )

            submit_button = st.form_submit_button(label="Aplicar")

        st.write("## Após os filtros")
        st.write(bank)
        st.write("Quantidade de linhas:", bank.shape[0])
        st.write("Quantidade de colunas:", bank.shape[1])

        col1, col2 = st.columns(spec=2)

        csv = df_to_csv(df=bank)
        col1.write("### Download CSV")
        col1.download_button(
            label="📥 Baixar como arquivo .csv",
            data=csv,
            file_name="df_csv.csv",
            mime="text/csv",
        )

        excel = df_to_excel(df=bank)
        col2.write("### Download Excel")
        col2.download_button(
            label="📥 Baixar como arquivo .xlsx",
            data=excel,
            file_name="df_excel.xlsx",
        )

        st.markdown("---")

        # Coluna 1
        bank_raw_target_pct = (
            bank_raw["y"].value_counts(normalize=True).to_frame() * 100
        )
        bank_raw_target_pct = bank_raw_target_pct.sort_index()
        # Coluna 2
        bank_target_pct = bank["y"].value_counts(normalize=True).to_frame() * 100
        bank_target_pct = bank_target_pct.sort_index()

        # PLOTS
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        if graph_type == "Barras":
            sns.barplot(
                x=bank_raw_target_pct.index,
                y="proportion",
                data=bank_raw_target_pct,
                ax=axes[0],
            )
            axes[0].bar_label(container=axes[0].containers[0])
            axes[0].set_title(label="Dados brutos", fontweight="bold")
            sns.barplot(
                x=bank_target_pct.index,
                y="proportion",
                data=bank_target_pct,
                ax=axes[1],
            )
            axes[1].bar_label(container=axes[1].containers[0])
            axes[1].set_title(label="Dados filtrados", fontweight="bold")
        else:
            bank_raw_target_pct.plot(
                kind="pie", autopct="%.2f", y="proportion", ax=axes[0]
            )
            axes[0].set_title("Dados brutos", fontweight="bold")
            bank_target_pct.plot(kind="pie", autopct="%.2f", y="proportion", ax=axes[1])
            axes[1].set_title("Dados filtrados", fontweight="bold")
        st.write("## Proporção de aceite")
        st.pyplot(plt)


if __name__ == "__main__":
    main()
