from fastapi import FastAPI
from fastapi.responses import JSONResponse
import face_recognition
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import pickle
import uvicorn
from pymongo import MongoClient
from bson import ObjectId
from urllib.parse import urlparse
import aiohttp

app = FastAPI()

mongo_uri = "mongodb://localhost:27017/"
db_name = "production"
client = MongoClient(mongo_uri)
db = client[db_name]
users_collection = db.users
mediaclients_collection = db['mediaclients']
client = ObjectId('675aa5756bdb6349ea4cecee')

with open("face_insight.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

async def download_image(url: str) -> np.ndarray:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to download image. HTTP status: {response.status}")
            
            image_data = await response.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            return image

@app.post("/recognize/")
async def recognize_faces(image_url: str):
    try:
        image = await download_image(image_url)
        result = []
    
        faces = app_face.get(image)
        
        if len(faces) == 0:
            return JSONResponse(content={"message": "No face detected"}, status_code=200)

        recognized_user_ids = set()

        for face in faces:
            face_embedding = face.embedding
            face_embedding = face_embedding / np.linalg.norm(face_embedding)

            cosine_similarities = np.dot(known_encodings, face_embedding)
            best_match_index = np.argmax(cosine_similarities)
            max_similarity = cosine_similarities[best_match_index]

            if max_similarity > 0.5:
                name = known_names[best_match_index]
                user_data = users_collection.find_one({"username": name})
                user_id = user_data["_id"] if user_data else None
            else:
                name = "Unknown"
                user_id = None
            if user_id:
                recognized_user_ids.add(user_id)
        if not recognized_user_ids:
            return JSONResponse(content={"message": "No recognized users found"}, status_code=200)
        
        results = mediaclients_collection.find_one({"_id": sas_obj_id})
        if not results:
            return JSONResponse(content={"message": "No matching media record found"}, status_code=404)
        
        albums = results.get('albums', [])
        matched_albums_info = []

        for album in albums:
            album_users = album.get('users', [])
            common_users = set(album_users).intersection(recognized_user_ids)
            if common_users:
                matched_albums_info.append({
                    "_id": str(album["_id"]),
                    "users": [str(uid) for uid in common_users]
                })
        
        final_result = {
            "img_url": image_url,
            "matched_albums": matched_albums_info
        }

        return JSONResponse(content=final_result, status_code=200)

    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
