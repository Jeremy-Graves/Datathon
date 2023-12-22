import os
import pandas as pd
from deepface import DeepFace

# Specify the directory containing the images
image_dir = '/home/jeremy/Desktop/datathon/faceimages'

# Specify the output CSV file
output_csv = '/home/jeremy/Desktop/datathon/face_features.csv'

# Functions to get age, gender, and race
def get_age(image_path):
    result = DeepFace.analyze(image_path, actions=['age'])
    return result[0]['age']

def get_gender(image_path):
    result = DeepFace.analyze(image_path, actions=['gender'])
    return result[0]['gender']

def get_race(image_path):
    result = DeepFace.analyze(image_path, actions=['race'])
    return result[0]['dominant_race']

# Initialize a list to store the results
results = []

# Analyze each image in the directory

for filename in os.listdir(image_dir):
    if filename.endswith('.png'):  
        try:
            print('Analyzing', filename)
            image_path = os.path.join(image_dir, filename)
            results.append({
                'filename': filename,
                'age': get_age(image_path),
                'gender': get_gender(image_path),
                'race': get_race(image_path),
            })
        except ValueError:
            print('Analyzing', filename)
            image_path = os.path.join(image_dir, filename)
            results.append({
                'filename': filename,
                'age': 'unknown',
                'gender': 'unknown',
                'race': 'unknown',
            })



# Convert the results to a DataFrame and write it to a CSV file
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)  # Use to_csv instead of write
