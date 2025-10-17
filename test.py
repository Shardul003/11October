import sqlite3

conn = sqlite3.connect("e_learning.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM courses")
rows = cursor.fetchall()

print("Courses:", rows)
conn.close()