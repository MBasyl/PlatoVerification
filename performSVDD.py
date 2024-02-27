from preprocessor import DataPreprocessor as dp
from models.SVDD import SVDDModel
import pandas as pd


# STEP 3 : Perform SVDD
csv_data = "AltParsed4kNewprofile.csv"  # from makedfprofiles.py
df = pd.read_csv(csv_data)
df = df[~df['title'].isin(
    ['Histories', 'Treatise', 'Histories#test', 'Treatise#test'])]

# Check class balance
print(df['binary_label'].value_counts())

# Define 'Plato' label
df['binary_label'] = df['title'].apply(
    lambda x: 1 if x in ['Law', 'Law#test', 'Late', 'Late#test'] else -1)  # removed: 'Late', 'Late#test'
svdd_model = SVDDModel()
svdd_model.perform_svdd(df,
                        params='SVDDAltParsed_LL46band03',
                        feature='word',
                        ngram=(4, 6),
                        bandwidth=0.3)
