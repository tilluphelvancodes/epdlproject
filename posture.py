import cv2
import numpy as np
from ultralytics import YOLO

# ============================= CONFIG ============================= #
NECK_THRESHOLD = 40
TORSO_THRESHOLD = 10
BAD_POSTURE_TIME = 180  # seconds

FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

# ============================= HELPERS ============================= #
def angle(p1, p2, p3):
    """
    Angle at p2 between p1-p2-p3
    """
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0
    cos = np.clip(dot / mag, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def vertical_point(p, offset=100):
    return np.array([p[0], p[1] - offset])

# ============================= MODEL ============================= #
model = YOLO("yolov8n-pose.pt")  # lightweight, fast

# ============================= MAIN ============================= #
cap = cv2.VideoCapture("input1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
bad_frames = 0
good_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    if not results or results[0].keypoints is None:
        cv2.imshow("Posture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    kpts = results[0].keypoints.xy.cpu().numpy()
    if len(kpts) == 0:
        continue

    kp = kpts[0]  # assume main person

    # Keypoints (COCO format)
    LEFT_EAR = kp[3]
    LEFT_SHOULDER = kp[5]
    LEFT_HIP = kp[11]

    # Compute angles
    neck_angle = angle(
        LEFT_EAR,
        LEFT_SHOULDER,
        vertical_point(LEFT_SHOULDER)
    )

    torso_angle = angle(
        LEFT_SHOULDER,
        LEFT_HIP,
        vertical_point(LEFT_HIP)
    )

    # Check posture
    good_posture = neck_angle < NECK_THRESHOLD and torso_angle < TORSO_THRESHOLD

    if good_posture:
        good_frames += 1
        bad_frames = 0
        color = GREEN
    else:
        bad_frames += 1
        good_frames = 0
        color = RED

    bad_time = bad_frames / fps

    # Draw keypoints and lines
    for point in [LEFT_EAR, LEFT_SHOULDER, LEFT_HIP]:
        cv2.circle(frame, tuple(point.astype(int)), 6, YELLOW, -1)

    cv2.line(frame, tuple(LEFT_EAR.astype(int)), tuple(LEFT_SHOULDER.astype(int)), color, 3)
    cv2.line(frame, tuple(LEFT_SHOULDER.astype(int)), tuple(LEFT_HIP.astype(int)), color, 3)

    # Display angles
    cv2.putText(
        frame,
        f"Neck: {int(neck_angle)}  Torso: {int(torso_angle)}",
        (10, 30),
        FONT, 0.9, color, 2
    )

    # Bad posture alert
    if bad_time > BAD_POSTURE_TIME:
        cv2.putText(
            frame,
            "⚠ BAD POSTURE ALERT",
            (10, 70),
            FONT, 1.0, RED, 3
        )

    cv2.imshow("Posture Detection (YOLOv8)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


