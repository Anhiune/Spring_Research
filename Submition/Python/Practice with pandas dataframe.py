import numpy as np
import pandas as pd

np.random.seed(42) #for reproducibility
dates = pd.date_range(start="2025-01-01", periods=30, freq='D')

data = {
    'date': dates,
    'anger': np.round(np.random.rand(30), 2),
    'anticipation': np.round(np.random.rand(30), 2),
    'disgust': np.round(np.random.rand(30), 2),
    'fear': np.round(np.random.rand(30), 2),
    'joy': np.round(np.random.rand(30), 2),
    'sadness': np.round(np.random.rand(30), 2),
    'surprise': np.round(np.random.rand(30), 2),
    'trust': np.round(np.random.rand(30), 2),
    'positive': np.round(np.random.uniform(0.5, 1.5, size=30), 2),
    'negative': np.round(np.random.uniform(0.2, 1.2, size=30), 2)
}

#Create data frame

mock_df = pd.DataFrame(data)

# Show the first 5 rows
print(mock_df.head(5))
#get information about the data frame (like type of the data in each column)
print(mock_df.info()) 
#
print(mock_df.describe())