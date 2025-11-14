# realtime_gesture_emotion.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
import time

# Try importing tensorflow/keras (may be large)
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    load_model = None
    print("Warning: TensorFlow/Keras not available (CNN won't run).", e)

# ----- Config -----
CNN_PATH = "models/cnn_gesture.h5"
RF_PATH = "models/rf_landmark.pkl"
RF_SCALER_PATH = "models/rf_scaler.pkl"
EMO_MODEL_PATH = "models/emotion_model.h5"
EMO_CLASSES_PATH = "models/emotion_classes.npy"
GESTURE_CLASSES_PATH = "models/gesture_classes.npy"  # optional: saved order of gesture classes

# Fusion weights (tune as needed)
W_CNN = 0.6
W_RF = 0.4

# CNN input size (must match how you trained your CNN)
CNN_IMG_SIZE = (128, 128)

# ----- Load models (defensive) -----
cnn_model = None
rf_model = None
rf_scaler = None
emotion_model = None
emotion_classes = None
gesture_classes = None

# load CNN
if load_model is not None and os.path.exists(CNN_PATH):
    try:
        cnn_model = load_model(CNN_PATH)
        print("Loaded CNN model:", CNN_PATH)
        # try to load gesture class names
        if os.path.exists(GESTURE_CLASSES_PATH):
            gesture_classes = np.load(GESTURE_CLASSES_PATH, allow_pickle=True)
            print("Loaded gesture classes from", GESTURE_CLASSES_PATH)
        else:
            # create placeholder names based on output size
            try:
                n_g = cnn_model.output_shape[-1]
                gesture_classes = np.array([str(i) for i in range(n_g)])
                print("Gesture classes inferred:", gesture_classes.shape[0], "classes")
            except Exception:
                gesture_classes = None
    except Exception as e:
        print("Failed to load CNN model:", e)

# load RF
if os.path.exists(RF_PATH):
    try:
        rf_model = joblib.load(RF_PATH)
        print("Loaded RandomForest:", RF_PATH)
    except Exception as e:
        print("Failed to load RF model:", e)
if os.path.exists(RF_SCALER_PATH):
    try:
        rf_scaler = joblib.load(RF_SCALER_PATH)
        print("Loaded RF scaler:", RF_SCALER_PATH)
    except Exception as e:
        print("Failed to load RF scaler:", e)

# load emotion model
if load_model is not None and os.path.exists(EMO_MODEL_PATH):
    try:
        emotion_model = load_model(EMO_MODEL_PATH)
        print("Loaded emotion model:", EMO_MODEL_PATH)
    except Exception as e:
        print("Failed to load emotion model:", e)
if os.path.exists(EMO_CLASSES_PATH):
    try:
        emotion_classes = np.load(EMO_CLASSES_PATH, allow_pickle=True)
        print("Loaded emotion classes:", list(emotion_classes))
    except Exception as e:
        print("Failed to load emotion classes:", e)

# ----- Mediapipe init -----
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True, min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Face landmark indices & feature fns (same as your emotion pipeline)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
NOSE_TIP = 1
CHIN = 152

def eye_aspect_ratio(pts, eye):
    A = np.linalg.norm(pts[eye[1]] - pts[eye[5]])
    B = np.linalg.norm(pts[eye[2]] - pts[eye[4]])
    C = np.linalg.norm(pts[eye[0]] - pts[eye[3]])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(pts, mouth):
    # safe indexing (some face mesh lists shorter), clip indices
    try:
        A = np.linalg.norm(pts[mouth[3]] - pts[mouth[9]])
        B = np.linalg.norm(pts[mouth[2]] - pts[mouth[10]])
        C = np.linalg.norm(pts[mouth[0]] - pts[mouth[6]])
        return (A + B) / (2.0 * C + 1e-6)
    except:
        return 0.0

def brow_distance(pts, brow, eye):
    brow_center = np.mean(pts[brow], axis=0)
    eye_center = np.mean(pts[eye], axis=0)
    return np.linalg.norm(brow_center - eye_center)

def face_tilt(pts):
    nose = pts[NOSE_TIP]
    chin = pts[CHIN]
    return np.arctan2(chin[1] - nose[1], chin[0] - nose[0])

# Hand helper: get bbox from 21 landmarks (normalized)
def hand_landmarks_to_bbox(landmarks, img_w, img_h, pad=0.2):
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    x1, x2 = max(0, int(min(xs))), min(img_w, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(img_h, int(max(ys)))
    # add padding
    w = x2 - x1; h = y2 - y1
    x1 = max(0, int(x1 - pad * w))
    y1 = max(0, int(y1 - pad * h))
    x2 = min(img_w, int(x2 + pad * w))
    y2 = min(img_h, int(y2 + pad * h))
    return x1, y1, x2, y2

# Flatten hand landmarks into feature vector (normalized by image size)
def flatten_hand_landmarks(landmarks):
    pts = []
    for lm in landmarks.landmark:
        pts.append(lm.x)
        pts.append(lm.y)
    return np.array(pts)  # length 42 (21*2)

# Softmax helper if needed
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ----- Webcam loop -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Starting real-time gesture+emotion (press ESC to quit).")
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== Face features & emotion prediction =====
    face_res = face_mesh.process(frame_rgb)
    emotion_label = None
    emotion_conf = 0.0
    if face_res.multi_face_landmarks:
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_res.multi_face_landmarks[0].landmark])
        leftEAR = eye_aspect_ratio(pts, LEFT_EYE)
        rightEAR = eye_aspect_ratio(pts, RIGHT_EYE)
        mar = mouth_aspect_ratio(pts, MOUTH)
        leftBrow = brow_distance(pts, LEFT_BROW, LEFT_EYE)
        rightBrow = brow_distance(pts, RIGHT_BROW, RIGHT_EYE)
        tilt = face_tilt(pts)
        feat = np.array([[leftEAR, rightEAR, mar, leftBrow, rightBrow, tilt]])
        if emotion_model is not None and 'scaler.pkl' in os.listdir('models') and os.path.exists("models/scaler.pkl"):
            try:
                scaler = joblib.load("models/scaler.pkl")
                feat_scaled = scaler.transform(feat)
            except Exception:
                feat_scaled = feat
        else:
            feat_scaled = feat
        if emotion_model is not None:
            try:
                preds = emotion_model.predict(feat_scaled, verbose=0)
                idx = int(np.argmax(preds))
                emotion_conf = float(np.max(preds))
                if emotion_classes is not None:
                    emotion_label = str(emotion_classes[idx])
                else:
                    emotion_label = str(idx)
            except Exception as e:
                # model predict error
                emotion_label = None

    # ===== Hand detection & gesture prediction =====
    hand_res = hands.process(frame_rgb)
    gesture_label = None
    gesture_conf = 0.0

    if hand_res.multi_hand_landmarks:
        # take first detected hand
        h_landmarks = hand_res.multi_hand_landmarks[0]
        # hand bbox and crop
        x1, y1, x2, y2 = hand_landmarks_to_bbox(h_landmarks, w, h, pad=0.35)
        # ensure non-empty crop
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            # CNN path
            p_cnn = None
            if cnn_model is not None:
                try:
                    crop_resized = cv2.resize(crop, CNN_IMG_SIZE)
                    crop_norm = crop_resized.astype(np.float32) / 255.0
                    p = cnn_model.predict(np.expand_dims(crop_norm, axis=0), verbose=0)[0]
                    # if model outputs logits instead of probs, convert
                    if np.any(p < 0) and np.max(p) > 1:
                        p = softmax(p)
                    p_cnn = p
                except Exception as e:
                    print("CNN predict error:", e)
            # RF path (landmark features)
            p_rf = None
            if rf_model is not None:
                try:
                    feats = flatten_hand_landmarks(h_landmarks).reshape(1, -1)
                    if rf_scaler is not None:
                        feats_scaled = rf_scaler.transform(feats)
                    else:
                        feats_scaled = feats
                    # rf.predict_proba
                    if hasattr(rf_model, "predict_proba"):
                        p_rf = rf_model.predict_proba(feats_scaled)[0]
                    else:
                        # fallback: one-hot via predict
                        pred = rf_model.predict(feats_scaled)[0]
                        classes = rf_model.classes_
                        p_rf = np.array([1.0 if c == pred else 0.0 for c in classes])
                except Exception as e:
                    print("RF predict error:", e)

            # Fuse predictions
            final_probs = None
            if p_cnn is not None and p_rf is not None:
                # need same class order / mapping. We'll try to align sizes:
                if p_cnn.shape == p_rf.shape:
                    final_probs = W_CNN * p_cnn + W_RF * p_rf
                else:
                    # fallback: prefer CNN if RF classes mismatch
                    final_probs = p_cnn
            elif p_cnn is not None:
                final_probs = p_cnn
            elif p_rf is not None:
                final_probs = p_rf

            if final_probs is not None:
                idx = int(np.argmax(final_probs))
                gesture_conf = float(np.max(final_probs))
                if gesture_classes is not None and idx < len(gesture_classes):
                    gesture_label = str(gesture_classes[idx])
                else:
                    gesture_label = str(idx)

            # Draw bbox and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,0), 2)
            lbl = (gesture_label if gesture_label is not None else "Unknown")
            cv2.putText(frame, f"Gesture: {lbl} ({gesture_conf:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

    # Display emotion
    if emotion_label is not None:
        cv2.putText(frame, f"Emotion: {emotion_label} ({emotion_conf:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # FPS
    now = time.time()
    fps = 1.0 / (now - last_time) if now != last_time else 0.0
    last_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Gesture + Emotion (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
