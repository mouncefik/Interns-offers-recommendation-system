from sqlalchemy import create_engine, Column, String 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

engine = create_engine('sqlite:///:memory:')
Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'
    id = Column(String, primary_key = True)
    name = Column(String)
    skills = Column(String)
    
class Offer(Base):
    __tablename__ =  'offers'
    id = Column(String, primary_key= True)
    name = Column(String)
    required_skills = Column(String)
 
Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)

students=[
    Student(id='s1', name='mouncef Ikhoubi', skills = 'Python, Machine Learning, Data Science'),
    Student(id='s2', name='Amine ', skills = 'Java, React, Cloud Computing '),
    Student(id='s3', name='Badr ', skills='Problem solving, Java, parallel computing, Cloud'),
    Student(id='s4', name='Alice Brown', skills='JavaScript, Front-end Development, UI/UX Design'),
    Student(id='s5', name='Mike Davis', skills='Python, Data Analysis, Business Intelligence')
]

offers = [
    Offer(id='o1', name='Data Scientist Expert', required_skills='Python, Machine Learning, Data Science'),
    Offer(id='o2', name='Web Developer Expert', required_skills='Java, Web Development, Cloud Computing'),
    Offer(id='o3', name='Game Developer Expert', required_skills='C++, Game Development, Computer Vision'),
    Offer(id='o4', name='Front-end Developer needed', required_skills='JavaScript, Front-end Development, UI/UX Design'),
    Offer(id='o5', name='Business Analyst +5 years experience', required_skills='Python, Data Analysis, Business Intelligence'),
    Offer(id='o6', name='AI Engineer urgent!', required_skills='Python, Machine Learning, Deep Learning'),
    Offer(id='o7', name='Cloud Architect Expert with 10 years of experience', required_skills='Java, Cloud Computing, DevOps'),
    Offer(id='o8', name='Cybersecurity Specialist', required_skills='C++, Cybersecurity, Networking')
]

session.add_all(students)
session.add_all(offers)
session.commit()

students_df = pd.read_sql(session.query(Student).statement, session.bind)
offers_df = pd.read_sql(session.query(Offer).statement, session.bind)

vectorizer = TfidfVectorizer()

students_skills = vectorizer.fit_transform(students_df['skills'])
offers_skills = vectorizer.transform(offers_df['required_skills'])

similarity_matrix = cosine_similarity(students_skills, offers_skills)

top_offers = []
for i in range(len(students_df)):
    top_indices = similarity_matrix[i].argsort()[-10:][::-1]
    top_offers.append(offers_df.iloc[top_indices]['name'].tolist())
    


results_df = pd.DataFrame({'Students': students_df['name'], 'Top Offers': top_offers})
results_df.to_csv('results.csv', index=False)
print(pd.read_csv('results.csv'))

session.close()

