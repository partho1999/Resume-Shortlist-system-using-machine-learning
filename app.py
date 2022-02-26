from fileinput import filename
from black import out
import matplotlib.colors as mcolors
import gensim
import gensim.corpora as corpora
from operator import index
from wordcloud import WordCloud
from pandas._config.config import options
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import Similar
from PIL import Image
import time
from docx import *
from docx import Document
from io import BytesIO
import shutil
import re
import os
import zipfile
from glob import glob
from fileReader import *




image = Image.open('Images/logo.png')
st.image(image, use_column_width=True)

st.title("Resume Matcher")

# Call fileRead
fileRead()
# Reading the CSV files prepared by the fileReader.py
Resumes = pd.read_csv('Resume_Data.csv')
Jobs = pd.read_csv('Job_Data.csv')


############################### JOB DESCRIPTION CODE ######################################
# Checking for Multiple Job Descriptions
# If more than one Job Descriptions are available, it asks user to select one as well.
if len(Jobs['Name']) <= 1:
    st.write(
        "There is only 1 Job Description present. It will be used to create scores.")
else:
    st.write("There are ", len(Jobs['Name']),
             "Job Descriptions available. Please select one.")


# Asking to Print the Job Desciption Names
if len(Jobs['Name']) > 1:
    option_yn = st.selectbox(
        "Show the Job Description Names?", options=['YES', 'NO'])
    if option_yn == 'YES':
        index = [a for a in range(len(Jobs['Name']))]
        fig = go.Figure(data=[go.Table(header=dict(values=["Job No.", "Job Desc. Name"], line_color='darkslategray',
                                                   fill_color='lightskyblue'),
                                       cells=dict(values=[index, Jobs['Name']], line_color='darkslategray',
                                                  fill_color='black'))
                              ])
        fig.update_layout(width=700, height=400)
        st.write(fig)


# Asking to chose the Job Description
index = st.slider("Which JD to select ? : ", 0,
                  len(Jobs['Name'])-1, 0)


option_yn = st.selectbox("Show the Job Description ?", options=['YES', 'NO'])
if option_yn == 'YES':
    st.markdown("---")
    st.markdown("### Job Description :")
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Job Description"],
                    fill_color='#f0a500',
                    align='center', font=dict(color='black', size=16)),
        cells=dict(values=[Jobs['Context'][index]],
                   fill_color='black',
                   align='left'))])

    fig.update_layout(width=800, height=500)
    st.write(fig)
    st.markdown("---")


#################################### SCORE CALCUATION ################################
@st.cache()
def calculate_scores(resumes, job_description):
    scores = []
    for x in range(resumes.shape[0]):
        score = Similar.match(
            resumes['TF_Based'][x], job_description['TF_Based'][index])
        scores.append(score)
    return scores


Resumes['Scores'] = calculate_scores(Resumes, Jobs)
########################################Sorting#################################################
Ranked_resumes = Resumes.sort_values(
    by=['Scores'], ascending=False).reset_index(drop=True)

Ranked_resumes['Rank'] = pd.DataFrame(
    [i for i in range(1, len(Ranked_resumes['Scores'])+1)])
#########################Export Resumes with Name, Rank, Email, Phone###########################
Email_lst=[]
Phone_lst=[]
for index,rows in Ranked_resumes.iterrows():
    nam = rows['Name']
    # print('Name: ',nam)
    try:
        my_text=Resumes[Resumes['Name']==nam].Context.tolist()[0]
        emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", str(my_text))[0]
        Email_lst.append(emails)
        phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', my_text)[0]
        Phone_lst.append(phone)
        # print('Text: ',my_text)
    except:
        Email_lst.append('Not Found')
        Phone_lst.append('Not Found')
        # print('Text: ',my_text)
    
output_df=Ranked_resumes[['Rank','Name','Scores']]
output_df['Email']=Email_lst
output_df['Phone']=Phone_lst
print(output_df.columns)
output_df = output_df.sort_values(
    by=['Scores'], ascending=False).reset_index(drop=True)

output_df['Rank'] = pd.DataFrame(
    [i for i in range(1, len(output_df['Scores'])+1)])
output_df.to_csv('shortlisted_list/ShortlistedResumes.csv',index=False)

#################################Save Shortlisted Resumes in New Folder########################
N=10 # Number of resumes, we want to shortlist
output_df=output_df[:N]
for inex,rows in output_df.iterrows():
    name=rows['Name']
    fileName=f'Data/Resumes/{name}'
    original=fileName
    target=f'Data/Shortlisted/{name}'
    shutil.copyfile(original, target)

    

    # document = opendocx(fileName)
    # body = document.xpath('/w:document/w:body', namespaces=nsprefixes)[0]
    # body.append(paragraph('Appending this.'))
#     with open(fileName, 'rb') as f:
#         source_stream = BytesIO(f.read())
#         document = Document(source_stream)
#         document.save(f'Data/Shortlisted/{name}')
# ###################################### SCORE TABLE PLOT ####################################

fig1 = go.Figure(data=[go.Table(
    header=dict(values=["Rank", "Name", "Scores"],
                fill_color='#00416d',
                align='center', font=dict(color='white', size=16)),
    cells=dict(values=[Ranked_resumes.Rank, Ranked_resumes.Name, Ranked_resumes.Scores],
               fill_color='black',
               align='left'))])

fig1.update_layout(title="Top Ranked Resumes", width=700, height=1100)
st.write(fig1)

@st.cache
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
##########################################################################################################

    
zipf = zipfile.ZipFile('python.zip','w')
os.chdir(f'Data/Shortlisted/')

for x in os.listdir():
    if x.endswith('pdf'):
        zipf.write(x, compress_type= zipfile.ZIP_DEFLATED)
    elif x.endswith('docx'):
        zipf.write(x, compress_type= zipfile.ZIP_DEFLATED)
zipf.close()

with open(r"C:\Users\Opus\Desktop\Naive-Resume-Matching-master\Python.zip", "rb") as fp:
    btn = st.download_button(
        label="Download Resumes",
        data= fp,
        file_name="myfile.zip",
        mime="application/x-7z-compressed"
    )
##########################################################################################################
st.markdown("---")

fig2 = px.bar(Ranked_resumes,
              x=Ranked_resumes['Name'], y=Ranked_resumes['Scores'], color='Scores',
              color_continuous_scale='haline', title="Score and Rank Distribution")
# fig.update_layout(width=700, height=700)
st.write(fig2)


st.markdown("---")

############################################ TF-IDF Code ###################################


@st.cache()
def get_list_of_words(document):
    Document = []

    for a in document:
        raw = a.split(" ")
        Document.append(raw)

    return Document


document = get_list_of_words(Resumes['Cleaned'])

id2word = corpora.Dictionary(document)
corpus = [id2word.doc2bow(text) for text in document]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, random_state=100,
                                            update_every=3, chunksize=100, passes=50, alpha='auto', per_word_topics=True)

################################### LDA CODE ##############################################


@st.cache  # Trying to improve performance by reducing the rerun computations
def format_topics_sentences(ldamodel, corpus):
    sent_topics_df = []
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df.append(
                    [i, int(topic_num), round(prop_topic, 4)*100, topic_keywords])
            else:
                break

    return sent_topics_df


################################# Topic Word Cloud Code #####################################
# st.sidebar.button('Hit Me')
st.markdown("## Topics and Topic Related Keywords ")
st.markdown(
    """This Wordcloud representation shows the Topic Number and the Top Keywords that contstitute a Topic.
    This further is used to cluster the resumes.      """)

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  collocations=False,
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
st.pyplot(plt)

st.markdown("---")

####################### SETTING UP THE DATAFRAME FOR SUNBURST-GRAPH ############################

df_topic_sents_keywords = format_topics_sentences(
    ldamodel=lda_model, corpus=corpus)
df_some = pd.DataFrame(df_topic_sents_keywords, columns=[
                       'Document No', 'Dominant Topic', 'Topic % Contribution', 'Keywords'])
df_some['Names'] = Resumes['Name']

df = df_some

st.markdown("## Topic Modelling of Resumes ")
st.markdown(
    "Using LDA to divide the topics into a number of usefull topics and creating a Cluster of matching topic resumes.  ")
fig3 = px.sunburst(df, path=['Dominant Topic', 'Names'], values='Topic % Contribution',
                   color='Dominant Topic', color_continuous_scale='viridis', width=800, height=800, title="Topic Distribution Graph")
st.write(fig3)


############################## RESUME PRINTING #############################

option_2 = st.selectbox("Show the Best Matching Resumes?", options=[
    'YES', 'NO'])
if option_2 == 'YES':
    indx = st.slider("Which resume to display ?:",
                     1, Ranked_resumes.shape[0], 1)

    st.write("Displaying Resume with Rank: ", indx)
    st.markdown("---")
    st.markdown("## **Resume** ")
    value = Ranked_resumes.iloc[indx-1, 2]
    st.markdown("#### The Word Cloud For the Resume")
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          colormap='viridis', collocations=False,
                          min_font_size=10).generate(value)
    plt.figure(figsize=(7, 7), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)

    st.write("With a Match Score of :", Ranked_resumes.iloc[indx-1, 6])
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Resume"],
                    fill_color='#f0a500',
                    align='center', font=dict(color='white', size=18)),
        cells=dict(values=[str(value)],
                   fill_color='black',
                   align='left'))])

    fig.update_layout(width=800, height=1200)
    st.write(fig)
    # st.text(df_sorted.iloc[indx-1, 1])
    st.markdown("---")

###########################Update Resume Data File with Email & Phone#####################
