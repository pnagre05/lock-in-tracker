# Focus Monitor Project Context

## Tech Stack
* Python 3
* Libraries: opencv-python, mediapipe, numpy, playsound==1.2.2

## Core Requirements
* Use mp.solutions.face_mesh.FaceMesh with refine_landmarks=True to ensure iris tracking is enabled.
* Sound playback MUST be handled via threading.Thread to prevent blocking the OpenCV video loop.

## Mathematical Constraints
To determine if a user is "focused," calculate the following:

1. Head Pose Estimation:
Use cv2.solvePnP to calculate head rotation.
Focus thresholds: abs(Yaw) < 15 degrees and abs(Pitch) < 15 degrees

2. Gaze Ratio (GR):
Calculate the horizontal position of the iris relative to the eye corners.
GR = distance(Iris_center, Inner_Corner) / distance(Outer_Corner, Inner_Corner)
Focus thresholds: GR between 0.35 and 0.65

3. Eye Aspect Ratio (EAR) (Optional for blink detection):
EAR = (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))

## App State Logic
* If Head Pose and Gaze Ratio are within thresholds -> is_focused = True
* If is_focused == False for > 2.0 consecutive seconds -> Trigger alert.mp3 via thread.