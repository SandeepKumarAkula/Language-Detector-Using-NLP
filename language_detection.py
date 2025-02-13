import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import pandas as pd

# Load dataset (Modify path if needed)
data_set = pd.read_csv("Language_Detection.csv")

# Filter unnecessary languages
language_column = "Language"
languages_to_drop = ["Kannada", "Sweedish", "Tamil", "Malayalam", "Greek", "Dutch", "Hindi"]
data_set = data_set[~data_set[language_column].isin(languages_to_drop)]

# Vectorization
count_vect = CountVectorizer(analyzer='word')
X = count_vect.fit_transform(data_set['Text'])  # Transform text into numerical form
y = data_set['Language']

# Train model
model = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
model.fit(X, y)

# Save the model and vectorizer
with open("language_model.pkl", "wb") as f:
    pickle.dump((model, count_vect), f)

print("Model trained and saved as language_model.pkl")
