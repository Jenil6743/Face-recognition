import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import os

def start_face_recognition(known_faces_pickle):
    """
    Real-time face recognition using webcam
    
    Args:
        known_faces_pickle: Path to pickle file containing face encodings
    """
    # Load known faces
    if not os.path.isfile(known_faces_pickle):
        print(f"Error: Pickle file not found: {known_faces_pickle}")
        return
        
    try:
        with open(known_faces_pickle, 'rb') as f:
            data = pickle.load(f)
            known_encodings = data['encodings']
            known_names = data['names']
        print(f"Loaded {len(known_names)} known faces")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting face recognition...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        
        for face in faces:
            if face.det_score < 0.5:
                continue
                
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            distances = [np.linalg.norm(embedding - known_emb) 
                       for known_emb in known_encodings]
            
            if distances:
                min_dist = min(distances)
                min_dist_idx = np.argmin(distances)
                
                if min_dist < 0.6: 
                    name = known_names[min_dist_idx]
                    confidence = 1 - (min_dist / 2) 
                else:
                    name = "Unknown"
                    confidence = 0
                
                bbox = face.bbox.astype(int)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), color, 2)
                
                text = f"{name} ({confidence:.2%})"
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"Detected: {name}, Distance: {min_dist:.3f}, "
                      f"Confidence: {confidence:.2%}")

        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_face_recognition("C:/Users/Janvi/OneDrive/Documents/face_encodings_insightfaces_v2.pkl")