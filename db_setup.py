import sqlite3
import pandas as pd
import os

# Paths to CSV files
courses_csv = os.path.join("data", "courses.csv")
students_csv = os.path.join("data", "students.csv")

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect("e_learning.db")
cursor = conn.cursor()

# Create courses table
cursor.execute("""
CREATE TABLE IF NOT EXISTS courses (
    CourseID TEXT PRIMARY KEY,
    Title TEXT,
    Category TEXT,
    Duration INTEGER
)
""")

# Create students table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    StudentID TEXT PRIMARY KEY,
    Name TEXT,
    Email TEXT,
    Country TEXT
)
""")

# Load CSVs using pandas
courses_df = pd.read_csv(courses_csv)
students_df = pd.read_csv(students_csv)

# Insert data into tables
courses_df.to_sql("courses", conn, if_exists="replace", index=False)
students_df.to_sql("students", conn, if_exists="replace", index=False)

# Confirm and close
conn.commit()
conn.close()

print("✅ Database setup complete. Tables 'courses' and 'students' are populated.")