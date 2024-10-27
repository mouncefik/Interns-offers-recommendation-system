from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use an in-memory SQLite database for demonstration
engine = create_engine('sqlite:///:memory:')
Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'
    id = Column(String, primary_key=True) # Add an ID column as primary key
    name = Column(String)
    skills = Column(String)

class Offer(Base):
    __tablename__ = 'offers'
    id = Column(String, primary_key=True) # Add an ID column as primary key
    name = Column(String)
    required_skills = Column(String)

Session = sessionmaker(bind=engine)
session = Session()

# Create the tables
Base.metadata.create_all(engine)

# Add some students and offers to the database
students = [
    Student(id='s1', name='John Doe', skills='Python, Machine Learning, Data Science'),
    Student(id='s2', name='Jane Smith', skills='Java, Web Development, Cloud Computing'),
    Student(id='s3', name='Bob Johnson', skills='C++, Game Development, Computer Vision'),
    Student(id='s4', name='Alice Brown', skills='JavaScript, Front-end Development, UI/UX Design'),
    Student(id='s5', name='Mike Davis', skills='Python, Data Analysis, Business Intelligence')
]

offers = [
    Offer(id='o1', name='Data Scientist', required_skills='Python, Machine Learning, Data Science'),
    Offer(id='o2', name='Web Developer', required_skills='Java, Web Development, Cloud Computing'),
    Offer(id='o3', name='Game Developer', required_skills='C++, Game Development, Computer Vision'),
    Offer(id='o4', name='Front-end Developer', required_skills='JavaScript, Front-end Development, UI/UX Design'),
    Offer(id='o5', name='Business Analyst', required_skills='Python, Data Analysis, Business Intelligence'),
    Offer(id='o6', name='AI Engineer', required_skills='Python, Machine Learning, Deep Learning'),
    Offer(id='o7', name='Cloud Architect', required_skills='Java, Cloud Computing, DevOps'),
    Offer(id='o8', name='Cybersecurity Specialist', required_skills='C++, Cybersecurity, Networking')
]

session.add_all(students)
session.add_all(offers)
session.commit()

# Query the students and offers data from the database
students_df = pd.read_sql(session.query(Student).statement, session.bind)
offers_df = pd.read_sql(session.query(Offer).statement, session.bind)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the skills data
students_skills = vectorizer.fit_transform(students_df['skills'])
offers_skills = vectorizer.transform(offers_df['required_skills'])

# Calculate the cosine similarity
similarity_matrix = cosine_similarity(students_skills, offers_skills)

# Get the top 3 offers for each student
top_offers = []
for i in range(len(students_df)):
    top_indices = similarity_matrix[i].argsort()[-3:][::-1]
    top_offers.append(offers_df.iloc[top_indices]['name'].tolist())

# Create a new dataframe with the students and their top 3 offers
results_df = pd.DataFrame({'Student': students_df['name'], 'Top Offers': top_offers})

# Print the results
print(results_df)

# Close the session
session.close()
