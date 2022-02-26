import streamlit as st
from PIL import Image
import os

def app():
    image = Image.open(r'D:\Naive-Resume-Matching-master\API\Images\logo.png')
    st.image(image, use_column_width=True)

    st.title("Upload Data")

    # st.write('This is the `upload_Data page` of this multi-page app.')

    # with open(os.path.join(r'D:\Naive-Resume-Matching-master\API\Data\Resumes',uploadedfile.name),"wb") as f:
    #      f.write(uploadedfile.getbuffer())

    uploaded_files= st.file_uploader("Upload Resumes:",type=["docx","doc","pdf","txt"], 
                                          accept_multiple_files = True)
    if uploaded_files is not None:
            # TO See details
            for image_file in uploaded_files:
                file_details = {"filename":image_file.name,"filetype":image_file.type,
                                "filesize":image_file.size}
                st.write(file_details)
                #st.image(load_image(image_file), width=250)
                
                #Saving upload
                with open(os.path.join(r"D:\Naive-Resume-Matching-master\API\Data\Resumes",image_file.name),"wb") as f:
                    f.write((image_file).getbuffer())
                
                st.success("File Saved")


    uploaded_files= st.file_uploader("Upload Job description:",type=["docx","doc","pdf","txt"], 
                                          accept_multiple_files = True)
    if uploaded_files is not None:
            # TO See details
            for image_file in uploaded_files:
                file_details = {"filename":image_file.name,"filetype":image_file.type,
                                "filesize":image_file.size}
                st.write(file_details)
                #st.image(load_image(image_file), width=250)
                
                #Saving upload
                with open(os.path.join(r"D:\Naive-Resume-Matching-master\API\Data\JobDesc",image_file.name),"wb") as f:
                    f.write((image_file).getbuffer())
                
                st.success("File Saved")

    