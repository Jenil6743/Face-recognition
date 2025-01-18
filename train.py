import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
from tqdm import tqdm

def train_face_recognition(train_dir, output_pickle_path, min_faces=10):
    """
    Train face recognition model using InsightFace and save embeddings to pickle file.
    
    Args:
        train_dir: Directory containing subdirectories of person images
        output_pickle_path: Path to save the pickle file
        min_faces: Minimum number of valid face detections required per person
    """
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    known_face_encodings = []
    known_face_names = []
    
    # Get list of subdirectories (one per person)
    person_dirs = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d))]
    
    print(f"Found {len(person_dirs)} persons in training directory")
    
    for person_name in person_dirs:
        person_dir = os.path.join(train_dir, person_name)
        face_encodings = []
        
        # Get all images for this person
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nProcessing {len(image_files)} images for {person_name}")
        
        # Process each image
        for img_file in tqdm(image_files):
            img_path = os.path.join(person_dir, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            # Get face detections
            faces = app.get(img)
            
            # Skip if no faces or multiple faces detected
            if len(faces) != 1:
                print(f"Skipping {img_file} - detected {len(faces)} faces")
                continue
            
            face = faces[0]
            
            # Skip if detection confidence is low
            if face.det_score < 0.5:
                print(f"Skipping {img_file} - low confidence: {face.det_score:.2f}")
                continue
            
            embedding = face.embedding
            face_encodings.append(embedding)
        
        if len(face_encodings) >= min_faces:
            average_embedding = np.mean(face_encodings, axis=0)
            average_embedding = average_embedding / np.linalg.norm(average_embedding)
            
            known_face_encodings.append(average_embedding)
            known_face_names.append(person_name)
            print(f"Added {person_name} with {len(face_encodings)} valid faces")
        else:
            print(f"Skipping {person_name} - only {len(face_encodings)} valid faces found")
    
    if known_face_encodings:
        data = {
            "encodings": known_face_encodings,
            "names": known_face_names
        }
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nSaved {len(known_face_names)} face encodings to {output_pickle_path}")
    else:
        print("No valid face encodings to save!")

if __name__ == "__main__":
    train_face_recognition(
        train_dir="C:/Users/Janvi/Desktop/Face_recog/train",
        output_pickle_path="face_insight.pkl",
        min_faces=10
    )