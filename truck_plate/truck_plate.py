import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

def detect_truck(img_path):
    img = cv2.imread(img_path)
    results = model(img, classes=[7])
    for r in results:
        coord = r.boxes.xyxy
        for c in coord:
            top_left = (int(c[0]), int(c[1]))
            bottom_right = (int(c[2]), int(c[3]))
            cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            cv2.imshow(f'Cropped Truck {top_left}', cropped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #cv2.imwrite('cropped_truck.jpg', cropped_img)
            
            
detect_truck('truck_data/truck_2.jpg')

            
