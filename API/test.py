import pathlib
import zipfile
from cv2 import dft
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


option = st.selectbox(
     'How would you like to be contacted?',
     ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)