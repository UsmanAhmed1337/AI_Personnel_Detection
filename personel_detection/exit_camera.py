import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import time
import sqlite3
from datetime import datetime

#DB and CV model
conn = sqlite3.connect('face_data.db')
cursor = conn.cursor()
model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(0)

#Variables for faces
known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_names = []

unknown_faces = 0
detected_persons = set()
entry_log_file = "logs/entry_log.txt"
exit_log_file = "logs/exit_log.txt"

#Load data from DB
def load_faces():
    cursor.execute('SELECT name, encoding FROM faces')
    rows = cursor.fetchall()
    for row in rows:
        name = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float64)
        known_face_names.append(name)
        known_face_encodings.append(encoding)
        print(known_face_names)
        print(known_face_encodings)
    return

#Detect any people in the frame
def person_detect(frame):
    results = model(frame, classes=[0])
    person_boxes = []
    for r in results:
        coord = r.boxes.xyxy
        for c in coord:
            top_left = (int(c[0]), int(c[1]))
            bottom_right = (int(c[2]), int(c[3]))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
            person_boxes.append((top_left, bottom_right))
    return frame, person_boxes

#Draw boxes around faces detected in the frame
def draw_facebox(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame

#Determine if the people detected face visible faces
def check_for_faces(frame, person_boxes, face_locations):
    for (p_top_left, p_bottom_right) in person_boxes:
        person_detected = False
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            if left >= p_top_left[0] and top >= p_top_left[1] and right <= p_bottom_right[0] and bottom <= p_bottom_right[1]:
                person_detected = True
                break
        if not person_detected:
            cv2.putText(frame, "Alert: Person detected but no face!", (p_top_left[0], p_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

#Checks a persons entry time and log their name and exit time 
def log_person(name):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    previous_time = None
    with open(entry_log_file, "r") as file:
        for line in file:
            log_name, log_time = line.strip().split(", ")
            if log_name == name:
                previous_time = log_time
    
    if previous_time:
        previous_time_obj = datetime.strptime(previous_time, "%Y-%m-%d %H:%M:%S")
        current_time_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        duration = (current_time_obj - previous_time_obj).total_seconds()
        with open(exit_log_file, "a") as new_file:
            new_file.write(f"{name}, {current_time}, {duration}s\n")
    else:
        with open(exit_log_file, "a") as file:
            file.write(f"{name}, {current_time}, no detected on entry\n")

#Self explanatory name
def save_unknown_face(face_encoding):
    global unknown_faces
    known_face_encodings.append(face_encoding)
    known_face_names.append(f"Unknown_{unknown_faces}")
    encoding_blob = np.array(face_encoding).tobytes()
    cursor.execute("REPLACE INTO faces (name, encoding) VALUES (?, ?)", (f"Unknown_{unknown_faces}", encoding_blob))
    conn.commit()
    unknown_faces += 1
    return

#Self explanatory name pt.2
def face_recognize(frame):
    #Resize and make image BW for faster inference
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    #Run inference
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    #Determine if faces are known or not
    detected_face_names = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        #Known
        if matches[best_match_index]: 
            name = known_face_names[best_match_index]
        #Unknown
        else: 
            cv2.putText(frame, "Unauthorized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            top *= 4
            left *= 4
            bottom *= 4
            right *= 4
            face_image = frame[top:bottom, left:right]
            save_unknown_face(face_encoding)

        #Check if person has already been logged in log
        detected_face_names.append(name)
        if name not in detected_persons and name != "Unknown":
            detected_persons.add(name)
            log_person(name)

    return frame, face_locations, detected_face_names

#Main function (no shit sherlock)
def main():
    load_faces()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame, person_boxes = person_detect(frame)
        if frame_count % 2 == 0:
            frame, face_locations, face_names = face_recognize(frame)
        frame = draw_facebox(frame, face_locations, face_names)
        frame = check_for_faces(frame, person_boxes, face_locations)
        frame_count += 1
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    return

main()