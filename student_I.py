import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the database file path
DB_FILE = 'student_offers.db'

# Check if the database file exists and delete it to start fresh
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

# Create a new SQLite database file
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()


# Sample offers data
offers = [
    ('Data Scientist Expert', 'Python, Machine Learning, Data Science'),
    ('Web Developer Expert', 'Java, Web Development, Cloud Computing'),
    ('Game Developer Expert', 'C++, Game Development, Computer Vision'),
    ('Front-end Developer', 'JavaScript, Front-end Development, UI/UX Design'),
    ('Business Analyst', 'Python, Data Analysis, Business Intelligence'),
    ('AI Engineer', 'Python, Machine Learning, Deep Learning'),
    ('Cloud Architect', 'Java, Cloud Computing, DevOps'),
    ('Cybersecurity Specialist', 'C++, Cybersecurity, Networking')
]

# Insert sample offers into the offers table
cursor.executemany('INSERT INTO offers (name, required_skills) VALUES (?, ?)', offers)
conn.commit()

# Function to suggest offers based on student skills
def suggest_offers(student_name, student_skills):
    # Create a DataFrame with student data
    student_df = pd.DataFrame({'name': [student_name], 'skills': [student_skills]})
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform student skills
    student_skills_vector = vectorizer.fit_transform(student_df['skills'])
    
    # Transform offer required skills
    offers_df = pd.read_sql('SELECT * FROM offers', conn)
    offers_skills_vector = vectorizer.transform(offers_df['required_skills'])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(student_skills_vector, offers_skills_vector)
    
    # Get top offers
    top_offers = []
    top_indices = similarity_matrix[0].argsort()[-3:][::-1]
    top_offers = [offers_df.iloc[i]['name'] for i in top_indices]
    
    return top_offers

# Get student input
student_name = input("Enter your name: ")
student_skills = input("Enter your skills (comma-separated): ")

# Add student to the students table
cursor.execute('INSERT INTO students (name, skills) VALUES (?, ?)', (student_name, student_skills))
conn.commit()

# Suggest offers to the student
top_offers = suggest_offers(student_name, student_skills)

print("Top offers for {} based on their skills:".format(student_name))
for offer in top_offers:
    print(offer)

# Close the connection
conn.close()