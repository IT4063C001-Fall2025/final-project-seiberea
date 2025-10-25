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

# In[7]:


import pandas as pd
import opendatasets as od

# Checkpoint 1 Data Sets:
od.download("https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships?select=Students+Social+Media+Addiction.csv")
od.download("https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health")
df = pd.read_excel("data_sources_standardized.xlsx")

# Checkpoint 2 Data Sets:
od.download("https://www.kaggle.com/datasets/shabdamocharla/social-media-mental-health")
od.download("https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health")

# 3 different datasets as well as 2 different data types. (Excel and CSV, as well as 2 different import methods, API and pandas read_excel function)


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# ## Checkpoint 2

# ## EDA
# 
# # Statisical Summaries

# In[ ]:


import pandas as pd

df = pd.read_csv(r"C:\Users\ellac\OneDrive\Desktop\Python Class\final-project-seiberea\social-media-and-mental-health\smmh.csv")

print("Columns:", df.columns.tolist())

rating_columns = [
    'Distraction_Rating',
    'Easily_Distracted_Rating',
    'Difficulty_Concentrating_Rating',
    'Frequency_Of_Comparing_Rating',
    'Depression_Frequency_Rating',
    'Trouble_Sleeping_Rating'
]

def central_tendencies(col):
    col = pd.to_numeric(col, errors='coerce')
    return {
        'mean': col.mean(),
        'median': col.median(),
        'mode': col.mode()[0] if not col.mode().empty else None
    }

for col in rating_columns:
    stats = central_tendencies(df[col])
    print(f"{col}:")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Mode:   {stats['mode']}")
    print("-" * 40)


# # Data Distributions

# In[ ]:


# Grouped by age and calculated the mean of Trouble_Sleeping_Rating
sleep_by_age = df.groupby('Age')['Trouble_Sleeping_Rating'].mean().reset_index()
print(sleep_by_age)


# # Feature Correlation

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\ellac\OneDrive\Desktop\Python Class\final-project-seiberea\social-media-and-mental-health\smmh.csv")

correlation_columns = [
    'Distraction_Rating',
    'Easily_Distracted_Rating',
    'Difficulty_Concentrating_Rating',
    'Frequency_Of_Comparing_Rating',
    'Depression_Frequency_Rating',
    'Trouble_Sleeping_Rating'
]

df[correlation_columns] = df[correlation_columns].apply(pd.to_numeric, errors='coerce')

correlation_matrix = df[correlation_columns].corr()

print("Correlations with Depression_Frequency_Rating:")
print(correlation_matrix['Depression_Frequency_Rating'].sort_values(ascending=False))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlations with Depression Frequency")
plt.tight_layout()
plt.show()


# # Detailed Write-Up (EDA)

# After calculating the statistical summary, I saw that most of the means and medians stayed right around the 2-3 mark, indicating moderate levels of all respective raitings. The modes however, were more spread out, possible indicating outliers. For the data distribution, I grouped by age and calculated the mean of Trouble_Sleeping_Rating. I expected to see the mean drop as the ages became older, however they tend to stay around roughly the same, with a few outliers here and there. I was extremely interested to see how each colum correlated with each other during the feature correlation section. While the obvious ones are correlated with each other, I was not expecting concentration difficulties to correlate closely to depression frequencey. I am interested to see if I can uncover more of that correlation as this project continues.
# 
# I encounted no data issues or data types that needed to be converted during this section.

# ## Data Visualizations

# In[37]:


# Histogram of Depression_Frequency_Rating grouped by Gender

sns.histplot(data=filtered_df, x='Depression_Frequency_Rating', hue='Gender', kde=True, multiple='stack')


# In[ ]:


# Violin Plot of Depression_Frequency_Rating by Age Group and Gender

bins = [13, 18, 22, 26, 30, 40, 60]
labels = ['13–17', '18–21', '22–25', '26–29', '30–39', '40+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

filtered_df = df[df['Gender'].isin(['Male', 'Female'])]

plt.figure(figsize=(10, 6))
sns.violinplot(
    data=filtered_df,
    x='Age_Group',
    y='Depression_Frequency_Rating',
    hue='Gender',
    split=True,
    palette='Set2'
)
plt.title('Depression Frequency by Age Group and Gender')
plt.xlabel('Age Group')
plt.ylabel('Depression Frequency Rating')
plt.tight_layout()
plt.show()


# In[ ]:


# Facet Grid of Depression_Frequency_Rating by Gender

g = sns.FacetGrid(filtered_df, col='Gender')
g.map(sns.histplot, 'Depression_Frequency_Rating', bins=5)


# # Detailed Write-Up (Data Visualizations)

# The first visualization I used was a heatmap, back during the feature correlation step. I used this visualization specifically because it would easily show correlations between all of the ratings, making it easy to link together ones who correlate with each other. The next visualization I used was a histogram. I used this as it provided an average (the midline) for the depression frequency on both male and female, giving an easy way for users to see average without having to calculate it. Next, I used a violin plot to show the depression frequency by age group and gender. By doing this, I was able to determine that females aged 18–25 reported higher depression frequency than males, with wider distributions and higher medians. I was able to figure this out by how violin charts are shown, the wider the shape, the more people gave that rating. Finally, I used a Facet Grid to show depression frequency by gender. While this is one of the simple visualizations I used, it's an easy way to see which group (gender in this case) has a more skewed set of data. As for this set, females are seen to report higher depression symptoms with a distribution skewed toward more frequent symptoms.

# # Data Cleaning and Transformations

# I did small amounts of cleaning and no transformations as I had no need to do that at this point in time. The only cleaning I did was renaming some of the important column headers in the smmh.csv dataset so coding scripts would be much easier. I only renamed ones that I deemed would be useful in analyzing data.

# # Machine Learning Plan

# This project lends itself well to supervised learning, especially classification. With thoughtful preprocessing and evaluation, I can build a model that predicts depression risk based on behavioral patterns. An issue that may arise could be some features may be highly correlated, which could lead to multicollinearity in regression models. The biggest challenge will be balancing predictive power with ethical responsibility and interpretability. It is extremely important to do this when creating a model with information like this.

# In[38]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

