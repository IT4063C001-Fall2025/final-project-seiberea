#!/usr/bin/env python
# coding: utf-8

# # The Effects of Social Media 📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# 
# Social Media and its Affect on Mental Health.
# 
# 
# Since the introduction of social media, patterens have been discovered that may correlate with reported mental health concerns in the youth population. Social media is now a staple in our daily lives and it's hard to come by someone who DOESN'T have any sort of social media account. Platforms like Tiktok, Twitter and Snapchat have all become way how we perceive the world around us and express ourselves. While on the surface it may seem harmless, studdies have shown that it just may do the exact opposite. This project aims to explore the depths of how social media may be the root cause of many mental health reports.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# 
# Does the amount of time spent on specific platforms correlate with anxiety or depression trends?
# 

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# A scatter plot showing average daily screen time vs. self-reported anxiety scores.
# - X-axis: Hours spent on social media per day
# - Y-axis: Anxiety score from a standardized survey.
# - Trend line: Positive slope could suggest higher screen time correlates with higher anxiety.
# 
# 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# Kaggle - Students' Social Media Addiction (https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships?select=Students+Social+Media+Addiction.csv)
# 
# GitHub - Mental Health Datasets (https://github.com/kharrigian/mental-health-datasets)
# 
# Kaggle - Social Media and Mental Health (https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health)

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[ ]:


import pandas as pd
import opendatasets as od
od.download("https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships?select=Students+Social+Media+Addiction.csv")
od.download("https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health")

df = pd.read_excel("data_sources_standardized.xlsx")

# 3 different datasets as well as 2 different data types. (Excel and CSV, as well as 2 different import methods, API and pandas read_excel function)


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

