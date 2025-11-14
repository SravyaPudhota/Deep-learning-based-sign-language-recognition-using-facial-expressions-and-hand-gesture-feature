import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

# Initialize FaceMesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Key landmark groups
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
NOSE_TIP = 1
CHIN = 152

# Helper functions
def eye_aspect_ratio(pts, eye):
    A = np.linalg.norm(pts[eye[1]] - pts[eye[5]])
    B = np.linalg.norm(pts[eye[2]] - pts[eye[4]])
    C = np.linalg.norm(pts[eye[0]] - pts[eye[3]])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(pts, mouth):
    A = np.linalg.norm(pts[mouth[3]] - pts[mouth[9]])  # vertical
    B = np.linalg.norm(pts[mouth[2]] - pts[mouth[10]])
    C = np.linalg.norm(pts[mouth[0]] - pts[mouth[6]])  # horizontal
    return (A + B) / (2.0 * C + 1e-6)

def brow_distance(pts, brow, eye):
    brow_center = np.mean(pts[brow], axis=0)
    eye_center = np.mean(pts[eye], axis=0)
    return np.linalg.norm(brow_center - eye_center)

def face_tilt(pts):
    nose = pts[NOSE_TIP]
    chin = pts[CHIN]
    return np.arctan2(chin[1] - nose[1], chin[0] - nose[0])

# Prepare data folder
os.makedirs("emotion_data", exist_ok=True)

# Choose emotion
emotion_label = input("Enter emotion label (happy/sad/angry/neutral/surprised): ").strip().lower()
output_csv = f"emotion_data/{emotion_label}.csv"

print(f"\nRecording '{emotion_label}' data — press ESC to stop...\n")

# Webcam
cap = cv2.VideoCapture(0)
records = []
start_time = time.time()

# EMA smoothing factor
alpha = 0.3
prev_features = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        pts = np.array([[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark])

        # Calculate features
        leftEAR = eye_aspect_ratio(pts, LEFT_EYE)
        rightEAR = eye_aspect_ratio(pts, RIGHT_EYE)
        mar = mouth_aspect_ratio(pts, MOUTH)
        leftBrow = brow_distance(pts, LEFT_BROW, LEFT_EYE)
        rightBrow = brow_distance(pts, RIGHT_BROW, RIGHT_EYE)
        tilt = face_tilt(pts)

        features = np.array([leftEAR, rightEAR, mar, leftBrow, rightBrow, tilt])

        # Apply smoothing
        if prev_features is not None:
            features = alpha * features + (1 - alpha) * prev_features
        prev_features = features

        elapsed = time.time() - start_time
        records.append([elapsed, *features, emotion_label])
        frame_count += 1

        # Display info
        cv2.putText(frame, f"Recording: {emotion_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Emotion Data Recording", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save or append data
df = pd.DataFrame(records, columns=["time", "leftEAR", "rightEAR", "MAR", "leftBrowDist", "rightBrowDist", "tilt", "emotion"])
if os.path.exists(output_csv):
    df_old = pd.read_csv(output_csv)
    df = pd.concat([df_old, df], ignore_index=True)

df.to_csv(output_csv, index=False)
print(f"✅ Saved {len(records)} new samples to {output_csv}")