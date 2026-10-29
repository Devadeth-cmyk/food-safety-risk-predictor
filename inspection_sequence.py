#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


DATA_PATH = "../data/processed/restaurant_identity.csv"

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# In[4]:


df.head()


# In[5]:


# Convert Inspection Date to datetime
df["Inspection Date"] = pd.to_datetime(
    df["Inspection Date"]
)

# Sort inspections chronologically for each restaurant
df = df.sort_values(
    ["Restaurant_ID", "Inspection Date"]
).reset_index(drop=True)

# Create inspection sequence starting from 0
df["Inspection_Sequence"] = (
    df.groupby("Restaurant_ID")
      .cumcount()
)

# Check the result
print(
    df[
        [
            "Restaurant_ID",
            "Inspection_Sequence",
            "Inspection Date",
            "Results"
        ]
    ].head(20)
)


# In[6]:


df.head(10)


# In[7]:


df.head(20)


# In[ ]:





# In[8]:


# Calculate the number of days since the previous inspection
df["Days_Since_Previous_Inspection"] = (
    df.groupby("Restaurant_ID")["Inspection Date"]
      .diff()
      .dt.days
)


# In[9]:


df = df.drop(columns=["Facility Type"])


# In[10]:


df.head(15)


# In[11]:


df = df.drop(
    columns=["Address", "Latitude_4dp","Longitude_4dp", "Latitude", "Longitude","License"]
)


# In[12]:


# OUTPUT_PATH = "../data/processed/restaurant_identity_added_feature.csv"

# df.to_csv(
#     OUTPUT_PATH,
#     index=False
# )

# print("File saved successfully!")
# print("Saved to:", OUTPUT_PATH)


# In[13]:


df['Results'].value_counts()

