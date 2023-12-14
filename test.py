import pickle
from joblib import dump

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Save the model using joblib
dump(model, 'model.joblib')
