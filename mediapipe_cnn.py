import cv2
import mediapipe as mp
import numpy as np
import time

# ============================
# Initialize MediaPipe Modules
# ============================
mp_hands = mp.solutions.hands
# Set max_num_hands=1 for single hand gesture detection
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
# Set max_num_faces=1 for single face expression analysis
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

# ============================
# Hand Gesture Detection Function (Updated with more signs)
# ============================
def detect_gesture(hand_landmarks):
    """
    Detects gestures based on the 'up' status of each finger.
    The finger status array is: [Thumb, Index, Middle, Ring, Pinky]
    """
    # 
    if not hand_landmarks:
        return "None"

    lm = hand_landmarks.landmark
    
    # List to store finger 'up' status (1 = up, 0 = down)
    finger_status = [0, 0, 0, 0, 0]

    # Thumb check (Vertical check: tip is higher than the IP joint)
    # Note: This simple Y-axis check can be reversed for the left hand.
    # For simplicity, we assume a static threshold.
    if lm[mp_hands.HandLandmark.THUMB_TIP].y < lm[mp_hands.HandLandmark.THUMB_IP].y:
        finger_status[0] = 1

    # Index, Middle, Ring, Pinky check (Vertical check: tip is higher than the PIP joint)
    for i, (tip_id, pip_id) in enumerate(zip([8, 12, 16, 20], [6, 10, 14, 18]), 1):
        if lm[tip_id].y < lm[pip_id].y:
            finger_status[i] = 1

    # --- Complex Gesture Checks ---
    
    # OK Sign 👌: Thumb & Index tips close
    thumb_tip_pos = np.array([lm[4].x, lm[4].y])
    index_tip_pos = np.array([lm[8].x, lm[8].y])
    distance = np.linalg.norm(thumb_tip_pos - index_tip_pos)
    
    # Check if OK sign shape is met AND other three fingers are up
    if finger_status == [0, 0, 1, 1, 1] and distance < 0.1: # Threshold is experimental
        return "OK Sign 👌"

    # --- Simple Finger Status Gesture Checks (Priority matters) ---

    # Thank You / Open Palm / Hello 🙏: All fingers up
    if finger_status == [1, 1, 1, 1, 1]:
        return "Thank You / Open Palm 🙏"

    # Emergency / I-Love-You Sign 🤟: Thumb, Index, Pinky up
    elif finger_status == [1, 1, 0, 0, 1]:
        return "Emergency / I Love You Sign 🤟"
        
    # Rock On 🤘: Index and Pinky up
    elif finger_status == [0, 1, 0, 0, 1]:
        return "Rock On 🤘"

    # L-Sign (Call Me) 🤙: Thumb and Index up
    elif finger_status == [1, 1, 0, 0, 0]:
        return "L-Sign (Call Me) 🤙"

    # Peace Sign ✌: Index and Middle up
    elif finger_status == [0, 1, 1, 0, 0]:
        return "Peace Sign ✌"
    
    # Thumbs Up 👍: Only Thumb up
    elif finger_status == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"

    # Pointing (Index) 👆: Only Index up
    elif finger_status == [0, 1, 0, 0, 0]:
        return "Pointing (Index) 👆"
    
    # Fist ✊: All fingers down
    elif finger_status == [0, 0, 0, 0, 0]:
        return "Fist ✊"
    
    return "Unknown"

# ============================
# Face Expression Analysis Function
# ============================
def analyze_face_features(face_landmarks):
    """
    Analyzes mouth and eyebrow landmarks to approximate a dominant expression.
    """
    # 
    if not face_landmarks:
        return "Neutral/No Face"

    lm = face_landmarks.landmark

    # 1. Mouth Feature (Smile/Sadness)
    # Left corner (291), Right corner (61), Lower lip center (14)
    y_corner_avg = (lm[291].y + lm[61].y) / 2
    y_center_lip = lm[14].y 
    mouth_curvature = y_center_lip - y_corner_avg

    if mouth_curvature > 0.02: # Corners pulled up
        return "Happy 😄"
    elif mouth_curvature < -0.01: # Corners pulled down
        return "Sad 😟"

    # 2. Eyebrow Feature (Surprise)
    # Inner eyebrow points (285, 55), Top of nose (5)
    y_eyebrow_avg = (lm[285].y + lm[55].y) / 2
    y_nose_top = lm[5].y

    eyebrow_height_diff = y_nose_top - y_eyebrow_avg
    
    if eyebrow_height_diff > 0.1: # Eyebrows raised
        return "Surprised 😮"
    
    return "Neutral"


# ============================
# Main Execution Block
# ============================

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

print("🔵 Starting camera... Press 'q' to quit.")

# Variables for displaying results
last_emotion = "Neutral/No Face"
last_gesture = "None"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed ended.")
        break

    frame = cv2.flip(frame, 1) # Flip horizontally for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- Hand Detection ---
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        last_gesture = detect_gesture(hand_landmarks)
    else:
        last_gesture = "None"

    # --- Face Expression Detection ---
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Draw the face mesh
        mp_draw.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS, 
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 150, 0), thickness=1))
        
        last_emotion = analyze_face_features(face_landmarks)
        
    else:
        last_emotion = "Neutral/No Face"

    # ----------------------------
    # Display Results
    # ----------------------------
    box_color = (0, 0, 0)
    text_color_gesture = (0, 255, 255) # Cyan
    text_color_emotion = (255, 200, 0) # Yellow-Blue
    
    # Draw background box for text
    cv2.rectangle(frame, (10, 10), (450, 130), box_color, -1)
    
    # Display Gesture
    cv2.putText(frame, f"Gesture: {last_gesture}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color_gesture, 2, cv2.LINE_AA)
    
    # Display Expression
    cv2.putText(frame, f"Expression: {last_emotion}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color_emotion, 2, cv2.LINE_AA)

    cv2.imshow("Live Gesture & Expression Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("🔴 Camera closed.")