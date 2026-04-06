import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ============================= CONFIG ============================= #
NECK_THRESHOLD  = 25   # degrees — forward head tilt limit
TORSO_THRESHOLD = 10   # degrees — spine lean limit
SMOOTH_FRAMES   = 8    # rolling average window to reduce flickering
BAD_POSTURE_TIME = 180  # seconds of continuous bad posture before alert

FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)
GRAY   = (180, 180, 180)

# ============================= HELPERS ============================= #
def angle(p1, p2, p3):
    """Angle at p2 in the p1-p2-p3 triplet."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))

def vertical_above(p, offset=100):
    return np.array([p[0], p[1] - offset])

def visible(p):
    return p[0] > 1 or p[1] > 1

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"

# ============================= MODEL ============================= #
model = YOLO("yolov8n-pose.pt")

# ============================= MAIN LOOP ============================= #
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

consecutive_bad  = 0
total_frames     = 0
total_bad_frames = 0
neck_buf  = deque(maxlen=SMOOTH_FRAMES)
torso_buf = deque(maxlen=SMOOTH_FRAMES)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    results = model(frame, verbose=False)

    # --- keypoint extraction ---
    if not results or results[0].keypoints is None:
        cv2.putText(frame, "No person detected", (10, 30), FONT, 0.8, GRAY, 2)
        cv2.putText(frame, "Press Q to end", (10, frame.shape[0] - 15), FONT, 0.6, WHITE, 1)
        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    kpts = results[0].keypoints.xy.cpu().numpy()
    if len(kpts) == 0:
        continue

    kp = kpts[0]  # COCO 17-keypoint format

    L_EAR      = kp[3]
    R_EAR      = kp[4]
    L_SHOULDER = kp[5]
    R_SHOULDER = kp[6]
    L_HIP      = kp[11]
    R_HIP      = kp[12]
    nose       = kp[0]

    l_body = visible(L_SHOULDER) and visible(L_HIP)
    r_body = visible(R_SHOULDER) and visible(R_HIP)

    if not l_body and not r_body:
        cv2.putText(frame, "No pose visible", (10, 30), FONT, 0.8, GRAY, 2)
        cv2.putText(frame, "Press Q to end", (10, frame.shape[0] - 15), FONT, 0.6, WHITE, 1)
        cv2.imshow("Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    use_left = l_body  # prefer left; fall back to right
    shoulder, hip = (L_SHOULDER, L_HIP) if use_left else (R_SHOULDER, R_HIP)
    ear_pt = L_EAR if use_left else R_EAR

    if visible(ear_pt):
        head = ear_pt
    elif visible(nose):
        head = nose
    else:
        head = None

    torso_buf.append(angle(shoulder, hip, vertical_above(hip)))
    if head is not None:
        neck_buf.append(angle(head, shoulder, vertical_above(shoulder)))

    torso_angle = float(np.mean(torso_buf))
    neck_angle  = float(np.mean(neck_buf)) if neck_buf else 0

    # --- posture judgement (on smoothed angles) ---
    neck_bad  = bool(neck_buf) and neck_angle >= NECK_THRESHOLD
    torso_bad = torso_angle >= TORSO_THRESHOLD
    good_posture = not neck_bad and not torso_bad

    if good_posture:
        consecutive_bad = 0
        color = GREEN
    else:
        consecutive_bad  += 1
        total_bad_frames += 1
        color = RED

    # --- draw skeleton (head→shoulder→hip) ---
    cv2.line(frame, tuple(shoulder.astype(int)), tuple(hip.astype(int)), color, 2)
    if head is not None:
        cv2.line(frame, tuple(head.astype(int)), tuple(shoulder.astype(int)), color, 2)
        cv2.circle(frame, tuple(head.astype(int)), 5, YELLOW, -1)
    cv2.circle(frame, tuple(shoulder.astype(int)), 5, YELLOW, -1)
    cv2.circle(frame, tuple(hip.astype(int)),      5, YELLOW, -1)

    # --- HUD ---
    neck_str = f"Neck:{int(neck_angle)}" if head is not None else "Neck:--"
    cv2.putText(frame, f"{neck_str}  Torso:{int(torso_angle)}",
                (10, 30), FONT, 0.75, color, 2)
    cv2.putText(frame, f"Session: {fmt_time(total_frames / fps)}",
                (10, 60), FONT, 0.65, WHITE, 1)
    cv2.putText(frame, "Press Q to end", (10, frame.shape[0] - 15), FONT, 0.6, WHITE, 1)

    if consecutive_bad / fps > BAD_POSTURE_TIME:
        cv2.putText(frame, "BAD POSTURE ALERT", (10, 95), FONT, 1.0, RED, 3)

    cv2.imshow("Posture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ============================= SUMMARY ============================= #
total_time   = total_frames / fps
bad_time     = total_bad_frames / fps
good_time    = total_time - bad_time
bad_pct      = (bad_time / total_time * 100) if total_time > 0 else 0

print("\n========== SESSION SUMMARY ==========")
print(f"  Total time      : {fmt_time(total_time)}")
print(f"  Good posture    : {fmt_time(good_time)}  ({100 - bad_pct:.1f}%)")
print(f"  Bad posture     : {fmt_time(bad_time)}  ({bad_pct:.1f}%)")
print("=====================================\n")

# Show summary in a window
summary = np.zeros((300, 500, 3), dtype=np.uint8)
lines = [
    ("SESSION SUMMARY", (WHITE, 1.0, 2)),
    (f"Total time   : {fmt_time(total_time)}", (WHITE, 0.75, 1)),
    (f"Good posture : {fmt_time(good_time)}  ({100 - bad_pct:.1f}%)", (GREEN, 0.75, 2)),
    (f"Bad posture  : {fmt_time(bad_time)}  ({bad_pct:.1f}%)", (RED, 0.75, 2)),
    ("Press any key to close", (GRAY, 0.6, 1)),
]
for i, (text, (clr, scale, thick)) in enumerate(lines):
    cv2.putText(summary, text, (30, 60 + i * 50), FONT, scale, clr, thick)

cv2.imshow("Summary", summary)
cv2.waitKey(0)
cv2.destroyAllWindows()
