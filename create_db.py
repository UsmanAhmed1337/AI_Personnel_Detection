import sqlite3

conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    name TEXT PRIMARY KEY,
    encoding BLOB
)
''')

conn.commit()