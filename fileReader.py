from operator import index
from pydoc import doc
from pandas._config.config import options
import Cleaner
import textract as tx
import pandas as pd
import os
import tf_idf
import re

def fileRead():
    resume_dir = "Data/Resumes/"
    job_desc_dir = "Data/JobDesc/"
    resume_names = os.listdir(resume_dir)
    job_description_names = os.listdir(job_desc_dir)

    document = []


    def read_resumes(list_of_resumes, resume_directory):
        placeholder = []
        for res in list_of_resumes:
            temp = []
            temp.append(res)
            text = tx.process(resume_directory+res, encoding='ascii')
            text = str(text, 'utf-8')
            temp.append(text)
            placeholder.append(temp)
        return placeholder


    document = read_resumes(resume_names, resume_dir)


    def get_cleaned_words(document):
        for i in range(len(document)):
            raw = Cleaner.Cleaner(document[i][1])
            document[i].append(" ".join(raw[0]))
            document[i].append(" ".join(raw[1]))
            document[i].append(" ".join(raw[2]))
            sentence = tf_idf.do_tfidf(document[i][3].split(" "))
            document[i].append(sentence)
        return document


    Doc = get_cleaned_words(document)


    Database = pd.DataFrame(document, columns=[
                            "Name", "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])
                            
    ##########################################Extract Email and phone number##################################################
    # Email_lst=[]
    # Phone_lst=[]
    # for index,rows in Database.iterrows():
    #     my_text = rows['Context']
    #     emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", str(my_text))[0]
    #     phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', my_text)[0]
    #     Email_lst.append(emails)
    #     Phone_lst.append(phone)
    # Database['Email']=Email_lst
    # Database['Phone']=Phone_lst

    # #########################################################################################################################
    Database.to_csv("Resume_Data.csv", index=False)

    # Database.to_json("Resume_Data.json", index=False)


    def read_jobdescriptions(job_description_names, job_desc_dir):
        placeholder = []
        for tes in job_description_names:
            temp = []
            temp.append(tes)
            text = tx.process(job_desc_dir+tes, encoding='ascii')
            text = str(text, 'utf-8')
            temp.append(text)
            placeholder.append(temp)
        return placeholder


    job_document = read_jobdescriptions(job_description_names, job_desc_dir)

    Jd = get_cleaned_words(job_document)

    jd_database = pd.DataFrame(Jd, columns=[
                            "Name", "Context", "Cleaned", "Selective", "Selective_Reduced", "TF_Based"])

    jd_database.to_csv("Job_Data.csv", index=False)
