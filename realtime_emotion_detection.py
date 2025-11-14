import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# === Load Model, Classes, and Scaler ===
model = load_model("emotion_model.h5")
classes = np.load("emotion_classes.npy", allow_pickle=True)
scaler = joblib.load("scaler.pkl")

# === Initialize FaceMesh ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# === Landmark Indices ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
NOSE_TIP = 1
CHIN = 152

# === Feature Functions ===
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

# === Webcam Capture ===
cap = cv2.VideoCapture(0)
last_emotion = None
smooth_confidence = 0

print("🎥 Real-time emotion detection started — press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        pts = np.array([[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark])

        # === Compute Features ===
        leftEAR = eye_aspect_ratio(pts, LEFT_EYE)
        rightEAR = eye_aspect_ratio(pts, RIGHT_EYE)
        mar = mouth_aspect_ratio(pts, MOUTH)
        leftBrow = brow_distance(pts, LEFT_BROW, LEFT_EYE)
        rightBrow = brow_distance(pts, RIGHT_BROW, RIGHT_EYE)
        tilt = face_tilt(pts)

        # === Prepare Feature Vector ===
        features = np.array([[leftEAR, rightEAR, mar, leftBrow, rightBrow, tilt]])
        features_scaled = scaler.transform(features)

        # === Predict Emotion ===
        preds = model.predict(features_scaled, verbose=0)
        emotion_idx = np.argmax(preds)
        confidence = preds[0][emotion_idx]
        emotion = classes[emotion_idx]

        # === Simple Temporal Smoothing ===
        if emotion == last_emotion:
            smooth_confidence = 0.8 * smooth_confidence + 0.2 * confidence
        else:
            smooth_confidence = 0.5 * confidence
        last_emotion = emotion

        # === Display Results ===
        cv2.putText(frame, f"Emotion: {emotion} ({smooth_confidence*100:.1f}%)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()