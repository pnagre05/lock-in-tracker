import cv2
import numpy as np
import threading
import time
import urllib.request
import os
import winsound
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# model download utility -------------------------------------------------------

def download_model_if_missing(model_path):
    """Download face_landmarker.task model from Google if it doesn't exist."""
    if os.path.exists(model_path):
        return
    
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    print(f"Downloading face_landmarker.task to {model_path}...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


# utility functions -----------------------------------------------------------

def distance(a, b):
    """Euclidean distance between two (x, y) points."""
    return np.linalg.norm(np.array(a, dtype="float32") - np.array(b, dtype="float32"))




# main application -----------------------------------------------------------

def main():
    # Download model if missing
    model_path = "face_landmarker.task"
    download_model_if_missing(model_path)
    
    cap = cv2.VideoCapture(0)
    
    # Initialize FaceLandmarker from Tasks API
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options)
    detector = vision.FaceLandmarker.create_from_options(options)

    # keep track of when distraction began
    distraction_start = None
    # allow pausing the evaluation
    is_paused = False
    # head-turn beep cooldown
    last_beep = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect face landmarks
        detection_result = detector.detect(mp_image)

        is_focused = False
        counter = 0.0
        # telemetry defaults
        yaw = 0.0
        pitch = 0.0
        gr = 0.0

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            # Convert normalized landmarks to pixel coordinates
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # *** gaze ratio (left eye) ***
            iris_center = pts[468]
            inner_corner = pts[133]
            outer_corner = pts[33]
            gr = distance(iris_center, inner_corner) / distance(outer_corner, inner_corner)

            # draw debug markers on eye
            cv2.circle(frame, iris_center, 3, (255, 0, 0), -1)
            cv2.circle(frame, inner_corner, 3, (0, 255, 0), -1)
            cv2.circle(frame, outer_corner, 3, (0, 0, 255), -1)

            # *** eye aspect ratio (left eye) ***
            # standard mediapipe left-eye indices used for EAR calculation
            p1 = pts[33]
            p2 = pts[160]
            p3 = pts[158]
            p4 = pts[133]
            p5 = pts[153]
            p6 = pts[144]
            ear = (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))

            # *** head pose estimation ***
            face_3d = np.array([
                (0.0, 0.0, 0.0),          # nose tip
                (0.0, -330.0, -65.0),     # chin
                (-225.0, 170.0, -135.0),  # left eye left corner
                (225.0, 170.0, -135.0),   # right eye right corner
                (-150.0, -150.0, -125.0), # left mouth corner
                (150.0, -150.0, -125.0)   # right mouth corner
            ])

            image_pts = np.array([
                pts[1],    # nose tip
                pts[199],  # chin
                pts[33],   # left eye left corner
                pts[263],  # right eye right corner
                pts[61],   # left mouth corner
                pts[291]   # right mouth corner
            ], dtype="double")

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vec, translation_vec = cv2.solvePnP(
                face_3d, image_pts, camera_matrix, dist_coeffs
            )
            rmat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rmat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            yaw = euler_angles[1][0]
            pitch = euler_angles[0][0]

            # split focus logic for debugging
            # new thresholds allow looking down; yaw +/-20, pitch [-25,10]
            is_head_focused = -20 < yaw < 20 and -25 < pitch < 10
            is_eye_focused = 0.35 < gr < 0.65
            is_focused = is_head_focused and is_eye_focused

            # beep on excessive yaw (head turn) with 1s cooldown
            nowt = time.time()
            if abs(yaw) > 20 and nowt - last_beep > 1.0:
                threading.Thread(target=winsound.Beep, args=(1000, 200), daemon=True).start()
                last_beep = nowt

            # telemetry overlay (always shown when landmarks present)
            cv2.putText(frame, f"Yaw: {int(yaw)}", (30, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"Pitch: {int(pitch)}", (30, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"GR: {gr:.2f}", (30, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # draw bounding box
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            box_color = (0, 255, 0) if is_focused else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # distraction timer logic
        now = time.time()
        if not is_focused and not is_paused:
            if distraction_start is None:
                distraction_start = now
            counter = now - distraction_start
            # previously played alert audio here; now just counting
        else:
            # either focused or paused -> reset timer
            distraction_start = None
            counter = 0.0

        # overlay counter
        cv2.putText(
            frame,
            f"Distraction: {counter:.1f}s",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            is_paused = not is_paused
        if is_paused:
            cv2.putText(frame, "PAUSED", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            # when paused we reset distraction timer and avoid sound
            distraction_start = None
        if key == 27:  # ESC to quit
            break
        cv2.imshow("Focus Monitor", frame)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
