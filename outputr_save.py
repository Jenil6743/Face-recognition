import cv2
import face_recognition
import pickle
import numpy as np

def recognize_faces(test_image_path, encodings_pickle="face_enc.pkl", tolerance=0.6):
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
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Detected face at (top={top}, right={right}, bottom={bottom}, left={left}) is: {name}")

    output_image_path = "recognized_faces.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved as {output_image_path}")

if __name__ == "__main__":
    test_image_path = r"C:/Users/Janvi/Desktop/DriveLabels/hostelwork/train_raw/rutvi/IMG20241213104352.jpg"
    recognize_faces(test_image_path, "face_enc.pkl", tolerance=0.6)
