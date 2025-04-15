import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import find_dotenv, load_dotenv
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from openai.embeddings_utils import get_embeddings
from embeddings_utils import get_embeddings
from sklearn.decomposition import PCA
import openai

import requests, json

  

load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

st.set_page_config(layout="wide")

st.title("Text Embeddings Visualization")

uploaded_file = st.file_uploader("Choose a file")

# TODO Check key before choosing file 

if uploaded_file:
    
    disable_button = True
    
    
    delimiter = st.text_input("Enter the delimiter", ";")
    drop_na = st.checkbox("Drop rows with N/A values")
    drop_duplicates = st.checkbox("Drop duplicate rows")
    df = pd.read_csv(uploaded_file, delimiter=delimiter)
    

    if drop_na:
        df = df.dropna()

    if drop_duplicates:
        df = df.drop_duplicates()

    columns = df.columns.tolist()
    
    dimensionslist = ['3', '8', '32']

    category_column = st.selectbox("Choose a column for categories:", columns)
    answer_column = st.selectbox("Choose a column for answers:", columns)
    dimension_str = st.selectbox("Choose a number of dimensions for answers:", dimensionslist, index=0)
    
    # Remove rows with empty answer_column
    df = df.loc[pd.notnull(df[answer_column])]
    df = df.reset_index(drop=True)

    df_filtered = df[[category_column, answer_column]].dropna()
    st.write(f"The DataFrame has {df_filtered.shape[0]} rows after filtering.")

    if st.button("Compute embeddings and plot", disabled=disable_button):
        
        # TODO - activate key here
        lic_function_results = "valid"
        
        print(lic_function_results)
  
        if "valid" in lic_function_results: 
            disable_button = False
            st.write("**:green[" + lic_function_results + "]**")

            #Indent start here for valid if
            
            #TODO - get top 50 records for starter license
        
            categories = sorted(df_filtered[category_column].unique())

            # TODO - select method here
            # return matrix
            api_type = os.getenv("OPENAI_API_TYPE", "openai")
            if api_type == "openai":
                matrix = get_embeddings(
                    df_filtered[answer_column].to_list(), engine="text-embedding-ada-002"
                )
            elif api_type == "azure":
                deployment_name = os.getenv("OPENAI_AZURE_DEPLOYMENT_NAME")
                embeddings = OpenAIEmbeddings(deployment=deployment_name)
                matrix = embeddings.embed_query(df_filtered[answer_column].to_list())

            dimension_int = int(dimension_str)
            pca = PCA(n_components=dimension_int)
            vis_dims = pca.fit_transform(matrix)
            df_filtered["embed_vis"] = vis_dims.tolist()

            cmap = px.colors.qualitative.Plotly
            fig = go.Figure()
            for i, cat in enumerate(categories):
                sub_matrix = np.array(
                    df_filtered[df_filtered[category_column] == cat]["embed_vis"].to_list()
                )
                x = sub_matrix[:, 0]
                y = sub_matrix[:, 1]
                z = sub_matrix[:, 2]
                answers = df_filtered[df_filtered[category_column] == cat][
                    answer_column
                ].tolist()
                color = cmap[i % len(cmap)]
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(color=color, size=6, opacity=0.8),
                        hovertemplate="%{text}",
                        hoverlabel=dict(font_size=16),
                        text=answers,
                        name=cat,
                    )
                )

            fig.update_layout(
                title="PCA of Text Embeddings Grouped by Categories",
                height=800,
                width=1200,
                scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            )

            col1, col2 = st.columns(2)
            col1.plotly_chart(fig, use_container_width=True)
            # TODO : show complete table with pages
            col2.dataframe(df_filtered.head(10))
            
            # TODO : Add option to export file
            
            # Add embeds to input file
            df1 = pd.DataFrame( vis_dims.tolist())
            
            #TODO - replace with column creator
            
            dim_col = []
            column_num = len(df1.columns)

            for x in range(column_num):
                dim_col.append('text_emb_dim' + str(x + 1) + '_' + answer_column)
            
            #df1.columns = ['text_emb_dim1_' + answer_column, 'text_emb_dim2_'+ answer_column, 'text_emb_dim3_'+ answer_column]
            df1.columns = dim_col
            
            results = df.join(df1)
            
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')


            csv = convert_df(results)
            
            embedded_file = "embeds.csv"
            results_file = "Textembeds_" + uploaded_file.name
            
            st.download_button(
                "Press to Download File with embeds(Page will be refreshed)"
                ,csv
                ,results_file
                ,"text/csv"
                ,key='download-csv'
            )
        else:
            st.write("**:red[" + lic_function_results + "]**")