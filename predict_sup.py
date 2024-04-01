import torch
import numpy as np
import cv2
from ultralytics import YOLO
from time import time
import supervision as sv


class ObjectDetection:
    
    def __init__(self, capture_index):
        
        self.capture_index = capture_index

        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1)
        

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)

        self.model = self.load_model()

        

    def load_model(self):

        model = YOLO('yolov8m.pt')
        model.fuse()

        return model
    
        

    def predict(self, img):

        results = self.model(img)[0]

        # Convert result to Supervision Detection object
        detections = sv.Detections.from_yolov8(results)

        return detections
    

    def plot_bboxes(self, detections, img):
        labels = [f"{self.load_model().model.names[class_id]} {confidence:0.2f}"
                  for _, confidence, class_id,_ in detections
                  ]
        
        # Add the box and tehe labels to the image
        annotated_image = self.box_annotator.annotate(scene=img, detections=detections, labels=labels)

        
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        frame_count = 0

        cv2.namedWindow('Intruder Detection', cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, img = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Prédiction sur l'image
                results = self.predict(img)
                if results:
                    img = self.plot_bboxes(results, img)
                    # Vérifiez si plot est une image valide
                    if img is not None and img.any():
                        cv2.imshow('Intruder Detection', img)
                    else:
                        print("No bounding boxes detected.")

                #cv2.imshow('Intruder Detection', plot)

                if cv2.waitKey(1) == 27:  # ESC key to break
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    obj_detection = ObjectDetection(0)  # 0 pour la webcam intégrée, ou spécifiez l'index de votre caméra
    obj_detection()
                    
                    