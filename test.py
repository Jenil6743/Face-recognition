import cv2
import face_recognition
import pickle
import numpy as np
from pymongo import MongoClient

def recognize_faces(test_image_path, encodings_pickle="face_enc.pkl", tolerance=0.6):
    mongo_uri = "mongodb://3.111.49.240:27017/"
    db_name = "production"
    client = MongoClient(mongo_uri)
    db = client[db_name]
    users_collection = db.users

    with open(encodings_pickle, "rb") as f:
        data = pickle.load(f)
    
    known_encodings = data["encodings"]
    known_names = data["names"]
    
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Could not load image from {test_image_path}")
        return
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_image, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if len(face_locations) == 0:
        print("No face detected")
        return

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_names[best_match_index]
            user_data = users_collection.find_one({"username": name})
            if user_data:
                user_id = user_data["_id"]
                print(f"User ID for {name}: {user_id}")
            else:
                print(f"{name} not found in the database.")
        
        print(f"Detected face at (top={top}, right={right}, bottom={bottom}, left={left}) is: {name}")
        
    client.close()

if __name__ == "__main__":
    test_image_path = r"C:/Users/Janvi/OneDrive/Pictures/urvish7.jpg"
    recognize_faces(test_image_path, "face_enc.pkl", tolerance=0.6)