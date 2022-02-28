import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import zipfile
from glob import glob
import plotly.graph_objects as go



def app():
    image = Image.open(r'D:\Naive-Resume-Matching-master\API\Images\logo.png')
    st.image(image, use_column_width=True)

    st.title("Download")

   

    output_df = pd.read_csv(r'D:\Naive-Resume-Matching-master\API\shortlisted_list\ShortlistedResumes.csv')

    
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=["Rank", "Name", "Scores","Email","Phone"],
                    fill_color='#00416d',
                    align='center', font=dict(color='white', size=16)),
        cells=dict(values=[output_df.Rank, output_df.Name, output_df.Scores, output_df.Email, output_df.Phone],
                fill_color='black',
                align='left'))])

    fig1.update_layout(title="Top Ranked Resumes", width=700, height=1100)
    st.write(fig1)

    #@st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(output_df)

    st.download_button(
    "Download The CSV",
    csv,
    "ShortlistedResumes.csv",
    "text/csv",
    key='download-csv'
    )
    def CreateZip():
        zipf = zipfile.ZipFile('python.zip','w')
        os.chdir(r'D:\Naive-Resume-Matching-master\API\Data\Shortlisted')

        for x in os.listdir():
            if x.endswith('pdf'):
                zipf.write(x, compress_type= zipfile.ZIP_DEFLATED)
            elif x.endswith('docx'):
                zipf.write(x, compress_type= zipfile.ZIP_DEFLATED)
        zipf.close()
    CreateZip()
    
    with open(r"D:\Naive-Resume-Matching-master\API\python.zip", "rb") as fp:
        btn = st.download_button(
            label="Download Resumes",
            data= fp,
            file_name="myfile.zip",
            mime="application/x-7z-compressed"
        )
    