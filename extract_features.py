import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

# === Mediapipe FaceMesh Setup ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True, min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# === Landmark Indices ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
NOSE_TIP = 1
CHIN = 152

# === Feature Extraction Functions ===
def extract_facemesh_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    h, w, _ = frame.shape
    landmarks = np.array([[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark])
    return landmarks

def eye_aspect_ratio(pts, eye):
    A = np.linalg.norm(pts[eye[1]] - pts[eye[5]])
    B = np.linalg.norm(pts[eye[2]] - pts[eye[4]])
    C = np.linalg.norm(pts[eye[0]] - pts[eye[3]])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(pts, mouth):
    A = np.linalg.norm(pts[mouth[3]] - pts[mouth[9]])
    B = np.linalg.norm(pts[mouth[2]] - pts[mouth[10]])
    C = np.linalg.norm(pts[mouth[0]] - pts[mouth[6]])
    return (A + B) / (2.0 * C + 1e-6)

def brow_distance(pts, brow, eye):
    brow_center = np.mean(pts[brow], axis=0)
    eye_center = np.mean(pts[eye], axis=0)
    return np.linalg.norm(brow_center - eye_center)

def face_tilt(pts):
    nose = pts[NOSE_TIP]
    chin = pts[CHIN]
    return np.arctan2(chin[1] - nose[1], chin[0] - nose[0])

# === Optional: Data Logging ===
SAVE_DATA = False  # set to True to record CSV data
emotion_label = "neutral"  # change this label per emotion while collecting data
output_folder = "emotion_data"
os.makedirs(output_folder, exist_ok=True)
data_buffer = []

# === Webcam Capture ===
cap = cv2.VideoCapture(0)
start_time = time.time()

print("🎥 Feature extraction started — press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pts = extract_facemesh_landmarks(frame)
    if pts is not None:
        leftEAR = eye_aspect_ratio(pts, LEFT_EYE)
        rightEAR = eye_aspect_ratio(pts, RIGHT_EYE)
        mar = mouth_aspect_ratio(pts, MOUTH)
        leftBrowDist = brow_distance(pts, LEFT_BROW, LEFT_EYE)
        rightBrowDist = brow_distance(pts, RIGHT_BROW, RIGHT_EYE)
        tilt = face_tilt(pts)

        # Display on screen
        cv2.putText(frame, f'L/R EAR: {leftEAR:.2f}/{rightEAR:.2f} | MAR: {mar:.2f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Brow(L/R): {leftBrowDist:.1f}/{rightBrowDist:.1f} | Tilt: {tilt:.2f}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Optionally save to buffer
        if SAVE_DATA:
            data_buffer.append([leftEAR, rightEAR, mar, leftBrowDist, rightBrowDist, tilt, emotion_label])

    cv2.imshow('Facial Feature Extraction (6 Features)', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Save Data if Recording ===
if SAVE_DATA and len(data_buffer) > 0:
    df = pd.DataFrame(data_buffer, columns=["leftEAR", "rightEAR", "MAR", "leftBrowDist", "rightBrowDist", "tilt", "emotion"])
    filename = os.path.join(output_folder, f"{emotion_label}_{int(time.time())}.csv")
    df.to_csv(filename, index=False)
    print(f"💾 Saved {len(df)} samples to {filename}")

cap.release()
cv2.destroyAllWindows()