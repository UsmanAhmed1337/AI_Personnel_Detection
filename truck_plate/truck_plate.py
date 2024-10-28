import cv2
import re
import easyocr
import time
from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')
license_model = YOLO('models/license_plate_detector.pt')
reader = easyocr.Reader(['en'])

def read_plate(frame):
    plates = license_model(frame, conf=0.6)
    for p in plates:
        coord = p.boxes.xyxy
        for c in coord:
            top_left = (int(c[0]), int(c[1]))
            bottom_right = (int(c[2]), int(c[3]))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv2.imwrite('truck_data/plate_image.jpg', cropped_frame)
            result = reader.readtext(cropped_frame, detail=0)
            numbers = [re.findall(r'\d+', r) for r in result]
            license_number = [num for sublist in numbers for num in sublist]
            if license_number:  
                return frame, license_number[0]
    return frame, None
    
def detect_truck(frame):
    results = model(frame, classes=[7], conf=0.5)
    for r in results:
        coord = r.boxes.xyxy
        for c in coord:
            top_left = (int(c[0]), int(c[1]))
            bottom_right = (int(c[2]), int(c[3]))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv2.imwrite('truck_data/truck_image.jpg', cropped_frame)
            frame_with_plate, license_number = read_plate(cropped_frame)
            if license_number:
                frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = frame_with_plate
                return frame, license_number 
        return frame, None

cap = cv2.VideoCapture('truck_data/1028.mp4')
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("truck_demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
extracted_number = None  
same_number = 0  
license_extracted = False
license_confirmed = False

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:  
        break
    if license_confirmed == False:
        frame, license_extracted = detect_truck(frame)
        if license_extracted:
            print(license_extracted)
            detected_number_str = ''.join(license_extracted)
            if detected_number_str == extracted_number:
                same_number += 1
            else:
                extracted_number = detected_number_str
                same_number = 1  
            if same_number >= 5:
                print(f"Confirmed license number: {extracted_number}")
                license_confirmed = True  
    else:
        cv2.putText(frame, f"Truck Liscense Plate Found : {extracted_number}", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Video", frame)
    video_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
inference_time = time.time() - start_time
print(inference_time)

            
