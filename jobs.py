# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:20:04 2021

@author: od297
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import string

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from gensim.models import Word2Vec
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')


import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')



st.set_page_config(page_title ='Indeed Data Science Job Postings')
st.header('Data Science Job Posts')
st.subheader('Indeed Dataset')

us_state_abbrev_reverse = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
}


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


#abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))

# Load CSV file into a DataFrame
ds_jobs_df = pd.read_csv('job_postings_June_2021.csv')

#drop duplicates
ds_jobs = ds_jobs_df.copy()
ds_jobs.drop_duplicates(inplace=True)
ds_jobs.dropna(inplace = True)


def clean_city(df):
    new = df['location'].str.split(",", expand = True)
    n1 = new[0].str.split("+", expand = True)
    new['city'] = n1[0]
    n4 = new['city'].str.split("•", expand = True)
    df['city'] = n4[0]
    #return job_data['city']

def clean_location(df):
    new = df['location'].str.split(",", expand = True)
    
    #clean_city(df)
    #Clean City
    n1 = new[0].str.split("+", expand = True)
    new['city'] = n1[0]
    n4 = new['city'].str.split("•", expand = True)
    new['city'] = n4[0]
    df['city'] = n4[0]
    
    #Clean State
    n2 = new[1].str.split(" ", expand = True)
    n2 = n2[1].str.replace('•', "+")
    n2 = n2.str.replace('+', " ")
    n2 = n2.str.split(" ", expand = True)
    new['State'] = n2[0]
    new['State'] = new['State'].replace(us_state_abbrev_reverse)
    new['State'].fillna(value='empty', inplace=True)
    new['State'].replace('empty', 'Same_with_City', inplace=True)
    
    new['State'] = np.where(new['State'] == 'Same_with_City', new['city'], new['State'])
    df['State'] = new['State']
    
    df['location'] = df[['city', 'State']].apply(lambda x: ', '.join(x), axis=1)
    
    return df

#Create a new dataframe
jobs = clean_location(ds_jobs)


def clean_text(text):
    text = str(text)
    text = text.replace("\n"," ")
    text = text.lower()
    #text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    return text.lower()

jobs["clean_description"] = jobs["description"].apply(clean_text)


Total_num_job_titles = jobs['title'].nunique()
#print('Total number of titles related to job postings: ', Total_num_job_titles)


#Keeping titles with Data Scientist/Data Science/Data Analyst/Data Analytics/Data Engineer
ds_jobs = ds_jobs[ds_jobs['title'].apply(lambda x: bool(re.findall('data analy\w*|data scien\w*|data engineer\w*', x.lower())))]

#Clean the job descriptions
ds_jobs["clean_description"] = ds_jobs["description"].apply(clean_text)

ds_jobs['description'] = ds_jobs['description'].apply(str.lower)



#Most common roles across companies
Most_common_job_title = jobs.groupby(['title'])['company'].count()
Most_common_job_title = Most_common_job_title.reset_index()
Most_common_job_title = Most_common_job_title.sort_values(['company'], ascending=False)
Most_common_job_title = Most_common_job_title.head(25)
Most_common_job_title

# Plot graph for top most offered roles
fig,ax=plt.subplots(figsize=(15,6))
ax=sns.barplot(x="title", y="company", data = Most_common_job_title)    
ax.set_xticklabels(Most_common_job_title['title'], rotation=90)
ax.set_xlabel('MOST WANTED JOB ROLES', fontsize=20, color='blue')
ax.set_ylabel('NO OF ROLES ACROSS INDUSTRY', fontsize=12,color='blue')


#There are so many job profiles in the given dataset so lets Categories them into 5; Data Scientist, Machine Learning Engineer, Data Analyst, Data Science Manager and Others

# Creating only 5 datascience roles among all
data = jobs.copy()

#data.dropna(subset=['title'], how='all', inplace = True)
data['position']= [x.upper() for x in data['title']]
data['description'] = [x.upper() for x in data['description']]

data.loc[data.position.str.contains("SCIENTIST"), 'position'] = 'Data Scientist'

data.loc[data.position.str.contains('ENGINEER'),'position']='Machine Learning Engineer'
data.loc[data.position.str.contains('PRINCIPAL STATISTICAL PROGRAMMER'),'position']='Machine Learning Engineer'

data.loc[data.position.str.contains('PROGRAMMER'),'position']='Machine Learning Engineer'
data.loc[data.position.str.contains('DEVELOPER'),'position']='Machine Learning Engineer'

data.loc[data.position.str.contains('ANALYST'), 'position'] = 'Data Analyst'
data.loc[data.position.str.contains('STATISTICIAN'), 'position'] = 'Data Analyst'

data.loc[data.position.str.contains('MANAGER'),'position']='Data Science Manager'

data.loc[data.position.str.contains('CONSULTANT'),'position']='Data Science Manager'
data.loc[data.position.str.contains('DATA SCIENCE'),'position']='Data Science Manager'
data.loc[data.position.str.contains('DIRECTOR'),'position']='Data Science Manager'

#data.position=data[(data.position == 'Data Scientist') | (data.position == 'Data Analyst') | (data.position == 'Machine Learning Engineer') | (data.position == 'Data Science Manager')]
#data.position=['Others' if x is np.nan else x for x in data.position]

position=data.groupby(['position'])['company'].count()   
position=position.reset_index(name='company')
position=position.sort_values(['company'],ascending=False)


t_roles = position.head(5)
t_roles.plot.bar(x='position', y='company')

data["clean_description"] = data["description"].apply(clean_text)


sentences = []
sent_word_sets = []
for row in data.iterrows():
    desc = row[1].clean_description
    word_tokens = nltk.word_tokenize(desc)
    sentences.append(word_tokens)
    sent_word_sets.append(set(word_tokens))
    

model = Word2Vec(sentences=sentences, window=5, min_count=10, workers=4)#,size=100


possible_words = set()
similar_words = model.wv.most_similar('bachelor', topn=30)
for tup in similar_words:
    possible_words.add(tup[0])
#similar_words


similar_words = model.wv.most_similar('masters', topn=30)
for tup in similar_words:
    possible_words.add(tup[0])
#similar_words

bachelor_list = ['bs','ba', 'bsc', 'babs', 'bsba', 'bsms', 'bsmsphd', 'bachelor', 'bachelors','undergraduate']
master_list = ['ma', 'ms', 'bsms', 'msc', 'msbs', 'msphd', 'bsmsphd', 'mdphd',
               'mba', 'phdms', 'master', 'masters', 'postgraduate']
phd_list = ['phd','bsmsphd','mdphd','msphd','doctoral','postgraduate','doctorate']


ds_jobs["sent_word_sets"] = sent_word_sets



def has_qual(sent_word_sets, description, qual_list):
    if qual_list == master_list or qual_list == phd_list:
        for word in qual_list:
            if word in sent_word_sets: #we want this part to be o(1) since qual_list is much shorter than word_set
                return True
            elif re.findall('advanced\s\w*\sdegree|graduate degree|advance\w* degree', description):
                return True 
        return False
    else:
        for word in qual_list:
            if word in sent_word_sets: #we want this part to be o(1) since qual_list is much shorter than word_set
                return True
            elif re.findall('4-year degree|university degree|undergraduate degree', description):
                return True 
        return False


ds_jobs["bachelors"] = ds_jobs.apply(lambda df: has_qual(df['sent_word_sets'], df['description'], bachelor_list), axis=1)
ds_jobs["masters"] = ds_jobs.apply(lambda df: has_qual(df['sent_word_sets'], df['description'], master_list), axis=1)
ds_jobs["phd"] = ds_jobs.apply(lambda df: has_qual(df['sent_word_sets'], df['description'], phd_list), axis=1)


print("Number of jobs with descriptions stating bachelors-like words/ expressions:",ds_jobs["bachelors"].sum())
print("Number of jobs with descriptions stating masters-like words/ expressions:",ds_jobs["masters"].sum())
print("Number of jobs with descriptions stating phd-like words/ expressions:",ds_jobs["phd"].sum())



def get_minimum(hasBsc,hasMsc,hasPhd):
    """
    returns minimum qualification if any
    """
    if hasBsc:
        return "Bachelors"
    
    elif hasMsc:
        return "Masters"
    
    elif hasPhd:
        return "Phd"
    
    else:
        return "No qualifications stated"



ds_jobs["min_qualification"] = ds_jobs.apply(lambda x: get_minimum(x.bachelors,x.masters,x.phd),axis=1)


df_degree_qualifications = ds_jobs["min_qualification"].value_counts().to_frame('Count')
df_degree_qualifications


print("The number jobs that require a minimum of Bachelors: ",df_degree_qualifications.loc["Bachelors", "Count"])
print("The number jobs that require a minimum of Masters: ",df_degree_qualifications.loc["Masters", "Count"])
print("The number jobs that require a minimum of Phd: ",df_degree_qualifications.loc["Phd", "Count"])
print("The number jobs that do not state education degree requirement: ",df_degree_qualifications.loc["No qualifications stated", "Count"])
print("The total number of jobs are: ",ds_jobs.shape[0])


fig,ax=plt.subplots(figsize=(15,6))
ax=sns.barplot(x=df_degree_qualifications.index, y="Count", data = df_degree_qualifications)    
ax.set_title('Degree Requirements in Data Jobs (Count)');



df_contain_degree = ds_jobs[ds_jobs['clean_description'].str.contains('degree')]

df_contain_else = ds_jobs[~ds_jobs['clean_description'].str.contains('degree')]

# jobs that have degree in description but did not state wht education level is required
set(df_contain_degree.index) - set(ds_jobs[~ds_jobs['min_qualification'].isin(['No qualifications stated'])].index)

# let's remove jobs 2 and 1972 and replace value in majors for job 1972 with not required
df_contain_degree.loc[2,'description_majors'] = ''
df_contain_degree.loc[1972,'description_majors'] = 'not required'

def degree_extraction(row):
    """
    Extract majors preferred using words following relevant mentionning of degree. 
    """
    return ' '.join([a for a in str(row).split('\n') if 'degree' in a.lower()])


def degree_extraction_other(row):
    """
    Extract majors preferred using words following relevant mentionning of abbreviated or non abbreviated 
    degree types.
    """
    regex_degrees = [r'\Ab\W*s\W*\s', r'\Ab\W*a\W*\s', r'\Am\W*s\W*\s', r'\Ab\W*s\W*\s',
                    r'\Ab\W*a\W*\s', r'\Aph\W*d\W*\s', r'master\'s|masters', r'doctorate|doctoral',
                    r'bachelor\'s|bachelors', r'education\w*\sbackground']
    majors_string = ''
    for expression in regex_degrees:
        try:
            majors_string += ' '.join([a for a in str(row).split('\n') if re.findall(expression, a)])
        except:
            pass
    return majors_string 


df_contain_degree['description_majors'] = df_contain_degree['description'].apply(degree_extraction).apply(str.lower)

df_contain_else['description_majors'] = df_contain_else['description'].apply(degree_extraction_other)

degree_df = pd.concat([df_contain_degree, df_contain_else])

#turn relevant majors into regex
relevant_degrees_dict = {'computer_science':'computer science|\Wcs\W|\Wce\W|computational science', 
                    'information_technology': 'information technology|information -technology|\Wit\W|information systems',
                    'mathematics':'mathematics',
                    'statistics': 'statistics|biostatistics|bioinformatics|informatics', 
                    'data_analytics': 'analytics|data analytics', 
                    'finance_accounting': 'financ\w+|accounting', 
                    'economics': 'economics|econometrics', 
                    'business': 'business|mba', 
                    'data_science': 'data science',
                    'engineering': 'engineering', 
                    'physical_sciences': 'physic\w+|astronomy|chemi\w+', 
                    'operations_research': 'operations research',
                    'life_sciences':'biology|plant science|biochemistry|biophysics',
                    'human_social_sciences': 'sociology|history|public policy|social science|politics|psychology'}


# get the number of mentions for each major and the set of jobs that have 
# specified majors. 
count_dict = {}
jobs_covered = set()

for degree_type in relevant_degrees_dict.keys():
    filter_condition = degree_df['description_majors'].apply(lambda x: bool(re.findall(
                                                                    relevant_degrees_dict[degree_type], x)))
    count_dict[degree_type] = filter_condition.sum()
    jobs_covered = jobs_covered | set(degree_df[filter_condition].index.to_list())
    
    
df_majors_count = pd.DataFrame.from_dict(count_dict, orient='index', columns=['Count']).sort_values(by="Count", ascending=False)
df_majors_count


fig,ax=plt.subplots(figsize=(15,6))
ax=sns.barplot(x=df_majors_count.index, y="Count", data = df_majors_count)
ax.set_title('Major Requirements in Data Jobs (Count)')
plt.xticks(rotation=70);


print(f'Number of jobs with at least one degree requirement: {len(jobs_covered)}')
print(f'Number of jobs with no degree specified/ for which we are unable to find the degrees: {len(degree_df.index) - len(jobs_covered)}')



st.text("1. For the degree requirements, 386 require a minimum of a Bachelor's degree, 199 a minimum of a Master's degree, and 22 a minimum of a PhD degree. A closer look at the 125 jobs for which we could not figure the minimum degree required reveals that 96 jobs contain at least one of the words (senior, principal, director, chief, head, lead, and staff) in the job title or specify at least one year of experience, showing that most of these jobs are not entry-level and so the degree requirement is not important., 2. Looking at the most popular majors, 310 jobs mention computer science among the majors preferred/ required, 292 mention statistics, 223 mention mathematics, 188 mention statistics, 127 mention data science, 117 mention economics, 79 mention the physical sciences, 75 mention business, 68 mention analytics or data analytics, 51 mention operations research, 46 mention IT, 31 mention finance/ accounting, 19 mention soaicl sciences, and 13 mention life sciences. A closer look at the 267 jobs for which we could not figure out the majors shows the following: - where a degree is mentioned, no specific details are given, but rather vagues words like quantitative field, stem field, related field, etc.; - 221 jobs contain at least one of the words (senior, principal, director, chief, head, lead, and staff) in the job title or specify at least one year of experience, showing that most of these jobs are not entry-level and so the major requirement may not be relevant.")



st.subheader('Kaggle Data')

df_2020 = pd.read_csv('kaggle_survey_2020_responses.csv')

columns_2020 = ['Q3', 'Q4', 'Q5', 'Q23_Part_1', 'Q23_Part_2', 'Q23_Part_3', 'Q23_Part_4', 'Q23_Part_5',
          'Q23_Part_6', 'Q37_Part_1', 'Q37_Part_2', 'Q37_Part_3', 'Q37_Part_4',
          'Q37_Part_5', 'Q37_Part_6', 'Q37_Part_7', 'Q37_Part_8', 'Q37_Part_9', 'Q37_Part_10']


df_2020_relevant = df_2020[columns_2020]

new_values_2020 = {"Analyze and understand data to influence product or business decisions": "analyze data", 
               "Build and/or run a machine learning service that operationally improves my product or workflows": "use machine learning",
               "Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data": "data infrastructure",
               "Build prototypes to explore applying machine learning to new areas": "build prototypes",
               "Experimentation and iteration to improve existing ML models": "improve machine learning",    
               "Do research that advances the state of the art of machine learning": "machine learning research",
               "None of these activities are an important part of my role at work": "none",
               "Other": "other"}


renames_2020 = {"Q3": "Country", "Q4": "FormalEducation", 
                "Q5": "CurrentJobTitle",
               "Q23_Part_1": "AnalyzeData", 
               "Q23_Part_2": "DataInfrastructure",
               "Q23_Part_3": "BuildPrototypes",
               "Q23_Part_4": "UseMachineLearning",
               "Q23_Part_5": "ImproveMachineLearning",
               "Q23_Part_6": "MachineLearningResearch",
               "Q9_Part_7": "NoneDataDuty",
               "Q9_Part_8": "OtherDataDuty", 
               "Q37_Part_1": "Coursera",
               "Q37_Part_2": "edX",
               "Q37_Part_3": "KaggleLearn",
               "Q37_Part_4": "DataCamp",
               "Q37_Part_5": "Fast.AI",
               "Q37_Part_6": "Udacity",
               "Q37_Part_7": "Udemy",
               "Q37_Part_8": "LinkedInLearning",
               "Q37_Part_9": "CloudCertifications", 
               "Q37_Part_10": "UniversityCourses",
               "Q37_Part_11": "NoneLearning"}


# rename columns and shorten job duty descriptions 
df_2020_relevant = df_2020_relevant.rename(columns = renames_2020)
df_2020_relevant = df_2020_relevant.replace(new_values_2020)

# drop questions row (i.e. row indexed at 0)
df_2020_relevant.drop(0, inplace=True)

# turn values under learning platforms in boolean
learning_columns = ["Coursera", "edX", "KaggleLearn",
                           "DataCamp", "Fast.AI", "Udacity",
                           "Udemy", "LinkedInLearning", "CloudCertifications", 
                           "UniversityCourses"]

df_2020_relevant[learning_columns] = df_2020_relevant[learning_columns].apply(lambda x: x.notnull())


# turn values of job duties into boolean
job_duties = ["AnalyzeData", "DataInfrastructure", "BuildPrototypes",
              "UseMachineLearning", "ImproveMachineLearning", "MachineLearningResearch"]

df_2020_relevant[job_duties] = df_2020_relevant[job_duties].apply(lambda x: x.notnull())


# shortern some values in education column 
education_dict = {
"Some college/university study without earning a bachelor’s degree": "Some college (no degree)",
}
df_2020_relevant['FormalEducation'] = df_2020_relevant['FormalEducation'].replace(education_dict)


kaggle_2020_data_clean = df_2020_relevant[df_2020_relevant['Country']=='United States of America']


kaggle_2020_data_clean['CurrentJobTitle'].value_counts()

#What is the Education Attainment breakdown of {insert job title(s)}s?
def visualize_education(df, job_title_list):
    """
    
    args:
    
    outputs:
    """
    df_occupation = df[df['CurrentJobTitle'].isin(job_title_list)]
    df_education = df_occupation['FormalEducation'].value_counts().to_frame()
    
    fig = df_education['FormalEducation'].plot(kind='pie',
                                    ylabel='',
                                    title='Education Attainment Breakdown',
                                    autopct="%.1f%%",
                                    figsize=(25,15),
                                    legend=True);
    
    return fig


data_professionals = ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'Business Analyst',
                     'Data Engineer', 'Statistician', 'DBA/Database Engineer']

df_copy = kaggle_2020_data_clean.copy()

visualize_education(df_copy, ['Data Scientist', 'Data Analyst', 'Data Engineer'])






























