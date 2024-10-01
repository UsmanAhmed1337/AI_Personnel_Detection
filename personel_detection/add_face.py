import numpy as np
import sqlite3
import face_recognition

conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()

#Encode face from image and store in DB
def encode_face(name, image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    face_encoding = face_encodings[0]
    encoding_blob = np.array(face_encoding).tobytes()
    cursor.execute("REPLACE INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_blob))
    conn.commit()

encode_face("Usman Ahmed", "face_data/Usman_Ahmed.jpg")
cursor.execute('SELECT * FROM faces')
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()





