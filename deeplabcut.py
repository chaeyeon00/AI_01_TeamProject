#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install deeplabcut
import deeplabcut
import os


# In[21]:


#프로젝트 생성
# C:\Users\User 경로에 생성
deeplabcut.create_new_project('D:/Python/ai_test_project','cykim',['D:/Python/AI_TESTDATSET/dog_1.mp4'],copy_videos=False,multianimal=False)


# In[22]:


config_path = 'D:/Python/ai_test_project-cykim-2022-06-19/config.yaml'


# In[23]:


deeplabcut.extract_frames(config_path)


# In[24]:


deeplabcut.check_labels(config_path)


# In[ ]:


## modelzoo 사용


# In[60]:


# stifle tensorflow warnings, like we get it already.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[77]:


import ipywidgets as widgets
from IPython.display import display

model_options = deeplabcut.create_project.modelzoo.Modeloptions
model_selection = widgets.Dropdown(
    options=model_options,
    value=model_options[0],
    description="Choose a DLC ModelZoo model!",
    disabled=False
)
display(model_selection)


# In[78]:


video_path = os.path.abspath('AI_TEST/catanddong.mp4')
#video_path = r'D:/Python/AI_TESTDATSET/dog_1.mp4'
print(video_path)
print(config_path)


# In[79]:


project_name = 'myDLC_modelZoo'
your_name = 'teamDLC'
model2use = model_selection.value
videotype = os.path.splitext(video_path)[-1].lstrip('.') #or MOV, or avi, whatever you uploaded!


# In[80]:


print(videotype)


# In[81]:


config_path, train_config_path = deeplabcut.create_pretrained_project(
    project_name,
    your_name,
    [video_path],
    videotype=videotype,
    model=model2use,
    analyzevideo=True,
    createlabeledvideo=True,
    copy_videos=True, #must leave copy_videos=True
)


# In[70]:


# Updating the plotting within the config.yaml file (without opening it ;):
edits = {
    'dotsize': 7,  # size of the dots!
    'colormap': 'spring',  # any matplotlib colormap!
    'pcutoff': 0.5,  # the higher the more conservative the plotting!
}
deeplabcut.auxiliaryfunctions.edit_config(config_path, edits)


# In[71]:


# re-create the labeled video (first you will need to delete in the folder to the LEFT!):
project_path = os.path.dirname(config_path)
full_video_path = os.path.join(
    project_path,
    'videos',
    os.path.basename(video_path),
)

#filter predictions (should already be done above ;):
deeplabcut.filterpredictions(config_path, [full_video_path], videotype=videotype)

#re-create the video with your edits!
deeplabcut.create_labeled_video(config_path, [full_video_path], videotype=videotype, filtered=True)

