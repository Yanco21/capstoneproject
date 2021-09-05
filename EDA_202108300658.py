#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

get_ipython().system('{sys.executable} -m pip install keyring artifacts-keyring')
get_ipython().system('{sys.executable} -m pip install --pre --upgrade --trusted-host pkgs.dev.azure.com --trusted-host pypi.org --trusted-host "*.blob.core.windows.net" --trusted-host files.pythonhosted.org --extra-index-url https://pkgs.dev.azure.com/mathco-products/_packaging/pip-codex-wf%40Local/pypi/simple/ "codex-widget-factory<=0.1"')


# In[1]:


# tags to identify this iteration when submitted
# example: codex_tags = {'env': 'dev', 'region': 'USA', 'product_category': 'A'}

codex_tags = {
}

from codex_widget_factory import utils
results_json=[]


# ### Ingestion File System

# In[2]:


from codex_widget_factory.ingestion.file_system import get_ingested_data as ingestion_file_system_get_ingested_data
response_0 = ingestion_file_system_get_ingested_data(file_path='Sales_Data.csv',
                        datasource_type='azure_blob_storage',
                        connection_uri='DefaultEndpointsProtocol=https;AccountName=coach;AccountKey=qtMAw9Z1T+1oOl+joMwfzdvnR0exMA4Qw50vniiaXOFvpOeFG7TP+g+DP4/iU7VKAUhirSpzrESvjF3U2Ld0KA==;EndpointSuffix=core.windows.net',
                        container_name='ddrs11/aegon_nischal')
results_json.append({
  'type': 'Ingestion',
  'name': 'File System',
  'component': 'get_ingested_data',
  'visual_results': utils.get_response_visuals(response_0),
  'metrics': False
})
utils.render_response(response_0)

#END WIDGET CODE


# In[4]:


import pandas as pd
import numpy as np


# ### Custom Data cleaning

# In[5]:


#BEGIN CUSTOM CODE BELOW...


# asking python to read the given values as missing values/ null values.
missing_value = ["nan","#DIV/0!",'Missing Values','Missing',' ','-','NA','N/A','NaN','na','#ref', np.nan] 
#response_0 = pd.read_csv('Sales_Data.csv', na_values = missing_value)
response_0 = response_0.replace(["nan","#DIV/0!",'Missing Values','Missing',' ','-','NA','N/A','NaN','na','#ref'],[np.nan]*11)

#Finding which columns has missing values
print('Columns and corresponding null values are:\n', response_0.isnull().sum().sort_values(ascending = False))

# Filling the NULL values in 'broker_percentage_lost_2yrs' with mean 
response_0['broker_percentage_lost_2yrs'].fillna(8.077246e+01,inplace = True)

# Finding mode values to fill na values in 'Industry' ,'Gen_Channel', 'MTP_Channel' columns
#I = sales['Industry'].mode()
G = response_0['Gen_Channel'].mode()
M = response_0['MTP_Channel'].mode()
P = response_0['Primary_Excess'].mode()
#Filling null values for 'Gen_Channel', 'MTP_Channel' & 'Primary_Excess'
response_0['Gen_Channel'].fillna('Non-MTP Intermediary',inplace = True)
response_0['MTP_Channel'].fillna('Non-MTP Brokers',inplace = True)
response_0['Primary_Excess'].fillna('Primary',inplace = True)

#Checking if any null values exist
print('After DATA CLEANING:\n The Sum of Null Values In The Data Set is:', response_0.isnull().sum().sum())
#dropping duplicate values
response_0=pd.concat([response_0["Opportunity_18_ID"].to_frame(),response_0.iloc[:,1:].drop_duplicates()],axis=1).dropna()

#changing the format of date column from string to datetime
response_0['Inception_Renewal_Date'] = pd.to_datetime(response_0['Inception_Renewal_Date'])

#upstream params found
# response_0

#put your output in this response param for connecting to downstream widgets
response_1 = response_0

#END CUSTOM CODE


# ### Transformation Remove Outliers

# In[6]:


#BEGIN WIDGET CODE BELOW...
# This function removes the outliers that are present in the column/dataset
#based on the interquartile range or replaces them with NaNs.
    

from codex_widget_factory.transformation.remove_outliers import get_transformed_data as transformation_remove_outliers_get_transformed_data
response_2 = transformation_remove_outliers_get_transformed_data(df = response_1, dep_var = 'Won',
                                                    col_names = ['gwp_lost_2yrs','gwp_won_2yrs','num_won_2yrs',
  'num_lost_2yrs','num_uniqprod_2yrs','num_won_2yrs',
 'num_lost_2yrs','num_wonnew_2yrs','num_quot_2yrs',
   'num_existing','broker_contact_num_won_2yrs',
  'broker_contact_days_since_last_lost','quot_not_won','num_lost_wo_subreceieved_2yrs',

'num_lost_postsub_2yrs',

'num_lost_postquot_2yrs',

'broker_days_since_last_won',

'broker_num_won_2yrs'])
results_json.append({
  'type': 'Transformation',
  'name': 'Remove Outliers',
  'component': 'get_transformed_data',
  'visual_results': utils.get_response_visuals(response_2),
  'metrics': False
})
utils.render_response(response_2)

#END WIDGET CODE



# ### Exploration Analysis Univariate (Numerical)

# In[7]:


#BEGIN WIDGET CODE BELOW...
response_num = response_1.select_dtypes(exclude = ['<M8[ns]',object]) 
from codex_widget_factory.exploration_analysis.univariate_report import summary as exploration_analysis_univariate_report_summary
response_2 = exploration_analysis_univariate_report_summary(response_num)
response_2= {
'Historam_Plots':response_2['Histogram Distribution']}
results_json.append({
  'type': 'Exploration Analysis',
  'name': 'Univariate Report',
  'component': 'summary',
  'visual_results': utils.get_response_visuals(response_2),
  'metrics': False
})
utils.render_response(response_2)

#END WIDGET CODE


# ### Custom Univariate (Categorical)

# In[8]:


cat = response_0.select_dtypes(include =[ object,'<M8[ns]'])
cat.pop('Opportunity_18_ID')
cat.pop('Entity_18_ID')
cat.pop('Intermediary_ID')
cat.pop('Intermediary_Key_Contact_18_ID')
cat
cat_var = [cat.columns[0:23]]
cat_var
list_1=['Business_Type', 'EntCountry', 'OppCountry', 'EntDivision',
        'OppDivision', 'Inception_Renewal_Date', 'Entity_Hierarchy_Level',
        'Entity_Record_Type', 'Industry', 'Line_of_Business', 'Policy_Type',
        'Primary_Excess', 'Product', 'Type', 'MTP_Channel', 'Gen_Channel', 
        'record_type', 'record_type_stage']

import plotly.express as px
for element in list_1:
    fig = px.histogram(cat, x =cat[element])
    fig.show()
    # response_1

#put your output in this response param for connecting to downstream widgets
#response_9

#END CUSTOM CODE



# ### Custom Bivariate

# In[9]:


import pandas as pd
import plotly.express as px
data1 = response_0.copy()
data1["temp"] = 1
#pd.options.display.float_format = '{:,.0%}'.format

def Percentage_Pivot(col,data):
    data_pivot = data.groupby([col,"Stage"], as_index = False)['temp'].sum()
    data_pivot1= data.groupby([col], as_index = False)['temp'].sum()
    data_pivot1 = data_pivot1.rename(columns = {"temp":"sum"})
    data_pivot = pd.merge(data_pivot,data_pivot1,on = col,how = "inner")
    data_pivot["Won"] = data_pivot["temp"]/data_pivot["sum"]
    return data_pivot

list_2=['Line_of_Business','Appetite',
        'Entity_Hierarchy_Level','Type','Entity_Record_Type',
        'gwp_lost_2yrs','gwp_won_2yrs','num_won_2yrs',
        'num_lost_2yrs','Gen_Channel','Product','num_uniqprod_2yrs',
        'Stage_Tracker_Length','num_won_2yrs',
        'num_lost_2yrs','num_wonnew_2yrs','num_quot_2yrs',
        'num_existing','broker_contact_num_won_2yrs',
        'broker_contact_days_since_last_lost','quot_not_won']

for element in list_2:
    temp_df = Percentage_Pivot(col = element, data = data1)
    print(temp_df)
    fig = px.bar(temp_df, x =  element,y= "Won", barmode = 'stack',color = "Stage")
    fig.show()
  
    #print(Percentage_Pivot)


# ### Custom Chi Square

# In[10]:


#BEGIN CUSTOM CODE BELOW...
df = response_1[['Industry','Appetite','Won',
                                        'Entity_Hierarchy_Level','Type',
                                                    'Entity_Record_Type','Gen_Channel','Product',
                                                    'quot_not_won','Stage_Tracker_Length','OppCountry','Business_Type']]


ordinal_label1 = {k: i for i, k in enumerate(df['Industry'].unique(), 0)}
ordinal_label2 = {k: i for i, k in enumerate(df['Entity_Hierarchy_Level'].unique(), 0)}
ordinal_label3 = {k: i for i, k in enumerate(df['Type'].unique(), 0)}
ordinal_label4 = {k: i for i, k in enumerate(df['Entity_Record_Type'].unique(), 0)}
ordinal_label5 = {k: i for i, k in enumerate(df['Gen_Channel'].unique(), 0)}
ordinal_label6 = {k: i for i, k in enumerate(df['Product'].unique(), 0)}
ordinal_label7 = {k: i for i, k in enumerate(df['quot_not_won'].unique(), 0)}
ordinal_label8 = {k: i for i, k in enumerate(df['Appetite'].unique(), 0)}
ordinal_label9 = {k: i for i, k in enumerate(df['OppCountry'].unique(), 0)}
ordinal_label10 = {k: i for i, k in enumerate(df['Business_Type'].unique(), 0)}
ordinal_label11 = {k: i for i, k in enumerate(df['Stage_Tracker_Length'].unique(), 0)}
df['Industry'] = df['Industry'].map(ordinal_label1)
df['Entity_Hierarchy_Level'] = df['Entity_Hierarchy_Level'].map(ordinal_label2)
df['Type'] = df['Type'].map(ordinal_label3)
df['Entity_Record_Type'] = df['Entity_Record_Type'].map(ordinal_label4)
df['Gen_Channel'] = df['Gen_Channel'].map(ordinal_label5)
df['Product'] = df['Product'].map(ordinal_label6)
df['quot_not_won'] = df['quot_not_won'].map(ordinal_label7)
df['Appetite'] = df['Appetite'].map(ordinal_label8)
df['OppCountry'] = df['OppCountry'].map(ordinal_label9)
df['Business_Type'] = df['Business_Type'].map(ordinal_label10)
df['Stage_Tracker_Length'] = df['Stage_Tracker_Length'].map(ordinal_label11)
#ordinal_label7


#upstream params found
# response_2

#put your output in this response param for connecting to downstream widgets
#response_15

#END CUSTOM CODE


# In[11]:


### train Test split is usually done to avaoid overfitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['Industry','Appetite',
                                                    'Entity_Hierarchy_Level','Type',
                                                    'Entity_Record_Type','Gen_Channel','Product',
                                                    'quot_not_won','Stage_Tracker_Length','OppCountry','Business_Type']],
                                              df['Won'],test_size=0.3,random_state=100)


# In[12]:


## Perform chi2 test
### chi2 returns 2 values
### Fscore and the pvalue
from sklearn.feature_selection import chi2
f_p_values=chi2(X_train,y_train)

    #df_temp = pd.DataFrame({"Column":[element],"t_stats":[t_value],"p_values":[p_value]})
   # df4 = pd.concat([df4,df_temp], axis = 0).reset_index(drop = True)


# In[13]:


f_values=pd.Series(f_p_values[0])
f_values.index=X_train.columns
f_values.sort_values(ascending = False)


# In[14]:


p_values=pd.Series(f_p_values[1])
p_values.index=X_train.columns

p_values


# ### Custom T Test

# In[15]:




from scipy.stats import ttest_ind
import plotly.express as px

list_3=[
'num_lost_wo_subreceieved_2yrs',

'num_lost_postsub_2yrs',

'num_lost_postquot_2yrs',

'num_won_2yrs',  'num_lost_2yrs',

'broker_days_since_last_won',

'broker_num_won_2yrs',

'broker_contact_num_won_2yrs',

'gwp_won_2yrs',  'gwp_lost_2yrs','num_uniqprod_2yrs'
]
df4 = pd.DataFrame()
count = 0

for element in list_3:
    count = count + 1
    df_lost = response_11[response_11['Won']==0]
    df_won = response_11[response_11['Won']==1]
    test_result = ttest_ind(df_won[element],df_lost[element])
    t_value = test_result[0]
    p_value = round(test_result[1],2)
    df_temp = pd.DataFrame({"Column":[element],"t_stats":[t_value],"p_values":[p_value]})
    df4 = pd.concat([df4,df_temp], axis = 0).reset_index(drop = True)  


# In[ ]:


response_1[list_3].corr()


# In[ ]:


df4


# In[ ]:





# ### Please save and checkpoint notebook before submitting params

# In[ ]:



currentNotebook = 'EDA_202108300658.ipynb'

get_ipython().system('jupyter nbconvert --to script {currentNotebook} ')


# In[ ]:



utils.submit_config_params(url='https://codex-api-stage.azurewebsites.net/codex-api/projects/upload-config-params/73E251D629FBDB78C0EF42C6C59E6CDFE2B849D098DF78A3B8AB286873DB1A7D', nb_name=currentNotebook, results=results_json, codex_tags=codex_tags, args={})

