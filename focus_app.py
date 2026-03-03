import cv2
import numpy as np
import threading
import time
import urllib.request
import os
import winsound
import queue
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk
import io
import sqlite3
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")



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


def cv2_to_pil(cv_image):
    """Convert cv2 BGR image to PIL RGB image."""
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image


# database helper ----------------------------------------------------------

def init_db(db_path='focus_data.db'):
    """Ensure the SQLite database and sessions table exist with proper schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            total_duration REAL,
            focused_duration REAL,
            final_score REAL,
            distraction_points INTEGER DEFAULT 0
        )
        '''
    )
    # migrate existing table to add distraction_points if missing
    try:
        cursor.execute("SELECT distraction_points FROM sessions LIMIT 1")
    except sqlite3.OperationalError:
        # column doesn't exist, add it
        cursor.execute("ALTER TABLE sessions ADD COLUMN distraction_points INTEGER DEFAULT 0")
    conn.commit()
    return conn


# camera capture thread ------------------------------------------------------

class CameraCapture(threading.Thread):
    """Continuously grab frames from the webcam, keeping only the latest one."""

    def __init__(self, src=0, queue_size=1):
        super().__init__(daemon=True)
        # use DirectShow backend on Windows to reduce latency and allow buffer control
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        # request only one frame in OpenCV internal buffer
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = threading.Event()
        self.running.set()

    def run(self):
        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            # always keep the newest frame; drop old if queue full
            try:
                self.queue.put(frame, block=False)
            except queue.Full:
                try:
                    _ = self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put(frame, block=False)
        self.cap.release()

    def stop(self):
        self.running.clear()


# backend tracking thread -------------------------------------------------------

class Backend(threading.Thread):
    """Runs face tracking in a background thread."""
    
    def __init__(self, model_path):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.queue = queue.Queue()
        self.pause_event = threading.Event()
        self.quit_event = threading.Event()
        self.distraction_start = None
        self.last_beep = 0
        self.last_distraction_increment = 0  # track when we last incremented distraction_points
        # dynamic thresholds (GUI updates these on-the-fly)
        self.yaw_threshold = 20.0
        self.pitch_threshold = 25.0
        # session timing
        self.total_session_time = 0.0
        self.cumulative_focused_time = 0.0
        self._last_time = time.time()
        # gamified tracking
        self.focus_streak = 0.0  # accumulates while focused
        self.distraction_points = 0  # increments on distraction events
        # capture thread that constantly reads from camera
        self.capture = CameraCapture()
        self.capture.start()
        
    def run(self):
        """Main tracking loop running in separate thread."""
        download_model_if_missing(self.model_path)
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options)
        detector = vision.FaceLandmarker.create_from_options(options)
        
        while not self.quit_event.is_set():
            try:
                frame = self.capture.queue.get_nowait()
            except queue.Empty:
                # no new frame yet, give CPU a moment
                time.sleep(0.005)
                continue
            
            # update session timing (ignore while paused)
            now = time.time()
            dt = now - self._last_time
            self._last_time = now
            if not self.pause_event.is_set():
                self.total_session_time += dt
            
            h, w = frame.shape[:2]
            # optionally downscale input to speed up Mediapipe processing
            if w > 640:
                scale = 640.0 / w
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                   interpolation=cv2.INTER_LINEAR)
                h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)
            
            is_focused = False
            counter = 0.0
            yaw = 0.0
            pitch = 0.0
            gr = 0.0
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
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
                p1 = pts[33]
                p2 = pts[160]
                p3 = pts[158]
                p4 = pts[133]
                p5 = pts[153]
                p6 = pts[144]
                ear = (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))
                
                # *** head pose estimation ***
                face_3d = np.array([
                    (0.0, 0.0, 0.0),
                    (0.0, -330.0, -65.0),
                    (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0),
                    (-150.0, -150.0, -125.0),
                    (150.0, -150.0, -125.0)
                ])
                
                image_pts = np.array([
                    pts[1], pts[199], pts[33], pts[263], pts[61], pts[291]
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
                
                # focus logic using dynamic threshold values
                is_head_focused = abs(yaw) < self.yaw_threshold and abs(pitch) < self.pitch_threshold
                is_eye_focused = 0.35 < gr < 0.65
                is_focused = is_head_focused and is_eye_focused
                
                # beep on excessive yaw triggers distraction point
                nowt = time.time()
                if abs(yaw) > 20 and nowt - self.last_beep > 1.0:
                    # beep + distraction increment (debounced)
                    if nowt - self.last_distraction_increment > 0.5:
                        self.distraction_points += 1
                        self.focus_streak = 0.0
                        self.last_distraction_increment = nowt
                    threading.Thread(target=winsound.Beep, args=(1000, 200), daemon=True).start()
                    self.last_beep = nowt
                
                # update focused time if not paused
                if not self.pause_event.is_set() and is_focused:
                    self.cumulative_focused_time += dt
                
                # draw bounding box
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                box_color = (0, 255, 0) if is_focused else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # gamified focus streak logic based on beeps
            now = time.time()
            is_paused = self.pause_event.is_set()
            
            if is_focused and not is_paused:
                # accumulate focus streak while focused
                self.focus_streak += dt
            # else: not focused - only beeps reset streak (no auto-timeout)
            
            # compute focus score
            focus_score = 0.0
            if self.total_session_time > 0:
                focus_score = (self.cumulative_focused_time / self.total_session_time) * 100
            # send data to queue
            self.queue.put({
                'frame': frame,
                'yaw': int(yaw),
                'pitch': int(pitch),
                'gr': float(gr),
                'focus_streak': float(self.focus_streak),
                'distraction_points': int(self.distraction_points),
                'is_focused': is_focused,
                'focus_score': focus_score
            })
        
        # stop camera capture thread and clean up
        self.capture.stop()
        detector.close()


# gui application -------------------------------------------------------

class FocusMonitorApp(ctk.CTk):
    """Modern dashboard GUI for focus monitor."""
    
    def __init__(self):
        super().__init__()
        self.title("Focus Monitor")
        self.geometry("1200x700")
        self.resizable(True, True)

        # database connection
        self.conn = init_db()

        # start backend
        self.backend = Backend("face_landmarker.task")
        self.backend.start()

        # configure root layout for tabview
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # create tab view
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew")
        self.tabview.add("Live Monitor")
        self.tabview.add("Analytics")
        live_tab = self.tabview.tab("Live Monitor")
        analytics_tab = self.tabview.tab("Analytics")

        # bind tab change event to refresh analytics
        try:
            self.tabview.bind("<<CTKTabviewTabChanged>>", self._on_tab_changed)
        except Exception:
            pass

        # -------- live monitor layout --------
        live_tab.grid_columnconfigure(0, weight=3)
        live_tab.grid_columnconfigure(1, weight=1)
        live_tab.grid_rowconfigure(0, weight=1)

        # video feed frame (with modern styling)
        self.video_frame = ctk.CTkLabel(live_tab, text="Loading...", fg_color="#1E1E1E", corner_radius=15)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_tk = None

        # sidebar in live tab (with modern styling)
        self.sidebar = ctk.CTkFrame(live_tab, fg_color="#1E1E1E", corner_radius=15)
        self.sidebar.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.sidebar.grid_rowconfigure(15, weight=1)

        # telemetry section
        ctk.CTkLabel(self.sidebar, text="Telemetry", font=("Arial", 16, "bold")).pack(pady=10)

        self.yaw_label = ctk.CTkLabel(self.sidebar, text="Yaw: --°", font=("Arial", 12))
        self.yaw_label.pack(pady=5)
        self.yaw_label.bind("<Enter>", lambda e: self.show_tooltip("Yaw", "Left/Right head rotation"))
        self.yaw_label.bind("<Leave>", lambda e: self.hide_tooltip())

        self.pitch_label = ctk.CTkLabel(self.sidebar, text="Pitch: --°", font=("Arial", 12))
        self.pitch_label.pack(pady=5)
        self.pitch_label.bind("<Enter>", lambda e: self.show_tooltip("Pitch", "Up/Down head tilt"))
        self.pitch_label.bind("<Leave>", lambda e: self.hide_tooltip())

        self.gr_label = ctk.CTkLabel(self.sidebar, text="GR: --.--", font=("Arial", 12))
        self.gr_label.pack(pady=5)
        self.gr_label.bind("<Enter>", lambda e: self.show_tooltip("GR", "Horizontal iris position (0.35-0.65)"))
        self.gr_label.bind("<Leave>", lambda e: self.hide_tooltip())

        # gamified distraction per minute display
        self.dpm_label = ctk.CTkLabel(self.sidebar, text="Distractions/Min: 0.0", font=("Arial", 12, "bold"), text_color="#00FF00")
        self.dpm_label.pack(pady=8)
        self.dpm_label.bind("<Enter>", lambda e: self.show_tooltip("Distractions/Min", "Distraction events per minute"))
        self.dpm_label.bind("<Leave>", lambda e: self.hide_tooltip())

        self.distraction_label = ctk.CTkLabel(self.sidebar, text="Distraction Points: 0", font=("Arial", 12, "bold"), text_color="#FF0000")
        self.distraction_label.pack(pady=8)
        self.distraction_label.bind("<Enter>", lambda e: self.show_tooltip("Points", "Distraction events detected"))
        self.distraction_label.bind("<Leave>", lambda e: self.hide_tooltip())

        # calibration sliders
        ctk.CTkLabel(self.sidebar, text="Yaw Threshold", font=("Arial", 12)).pack(pady=(10,2))
        self.yaw_slider = ctk.CTkSlider(self.sidebar, from_=10, to=45, number_of_steps=35,
                                        command=self._update_yaw_threshold)
        self.yaw_slider.set(self.backend.yaw_threshold)
        self.yaw_slider.pack(pady=5, fill="x", padx=5)

        ctk.CTkLabel(self.sidebar, text="Pitch Threshold", font=("Arial", 12)).pack(pady=(10,2))
        self.pitch_slider = ctk.CTkSlider(self.sidebar, from_=10, to=45, number_of_steps=35,
                                          command=self._update_pitch_threshold)
        self.pitch_slider.set(self.backend.pitch_threshold)
        self.pitch_slider.pack(pady=5, fill="x", padx=5)

        # focus score label
        self.score_label = ctk.CTkLabel(self.sidebar, text="Score: --%", font=("Arial", 12, "bold"), text_color="#00CCFF")
        self.score_label.pack(pady=10)

        # tooltip info label
        self.info_label = ctk.CTkLabel(self.sidebar, text="", font=("Arial", 10), text_color="cyan", wraplength=120)
        self.info_label.pack(pady=20, padx=5)

        # buttons (with modern styling)
        self.pause_button = ctk.CTkButton(self.sidebar, text="Pause", command=self.toggle_pause, font=("Arial", 12), corner_radius=10, fg_color="#007ACC")
        self.pause_button.pack(pady=10, fill="x", padx=5)

        self.quit_button = ctk.CTkButton(self.sidebar, text="Quit", command=self.quit_app, font=("Arial", 12), fg_color="#CC0000", corner_radius=10)
        self.quit_button.pack(pady=10, fill="x", padx=5)

        # -------- analytics tab setup --------
        # split analytics into two frames
        analytics_tab.grid_rowconfigure(0, weight=0)
        analytics_tab.grid_rowconfigure(1, weight=1)
        analytics_tab.grid_columnconfigure(0, weight=1)

        # top frame: lifetime stats
        stats_frame = ctk.CTkFrame(analytics_tab, fg_color="#1E1E1E", corner_radius=15)
        stats_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        stats_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(stats_frame, text="Lifetime Stats", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=10)

        self.total_hours_label = ctk.CTkLabel(stats_frame, text="Total Hours Focused\n0.0h", font=("Arial", 12, "bold"), text_color="#00FF00")
        self.total_hours_label.grid(row=1, column=0, padx=10, pady=10)

        self.total_distractions_label = ctk.CTkLabel(stats_frame, text="Total Distractions\n0", font=("Arial", 12, "bold"), text_color="#FF0000")
        self.total_distractions_label.grid(row=1, column=1, padx=10, pady=10)

        self.avg_score_label = ctk.CTkLabel(stats_frame, text="All-Time Avg Score\n0.0%", font=("Arial", 12, "bold"), text_color="#00CCFF")
        self.avg_score_label.grid(row=1, column=2, padx=10, pady=10)

        # bottom frame: trends graph
        graph_frame = ctk.CTkFrame(analytics_tab, fg_color="#1E1E1E", corner_radius=15)
        graph_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        graph_frame.grid_rowconfigure(0, weight=1)
        graph_frame.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('#1E1E1E')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1E1E1E')
        # style axes for dark theme
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.refresh_stats()
        self.refresh_plot()

        # polling loop
        self.after(30, self.update_gui)
    
    def _update_yaw_threshold(self, value):
        self.backend.yaw_threshold = float(value)

    def _update_pitch_threshold(self, value):
        self.backend.pitch_threshold = float(value)

    def _on_tab_changed(self, event):
        # refresh analytics whenever user switches to that tab
        if self.tabview.get() == "Analytics":
            self.refresh_stats()
            self.refresh_plot()

    def refresh_stats(self):
        """Update lifetime stats labels from database."""
        cursor = self.conn.cursor()
        try:
            # total focused duration
            cursor.execute("SELECT SUM(focused_duration) FROM sessions")
            total_focused = cursor.fetchone()[0] or 0.0
            hours = total_focused / 3600.0

            # total distraction points
            cursor.execute("SELECT SUM(distraction_points) FROM sessions")
            total_distractions = cursor.fetchone()[0] or 0

            # average score
            cursor.execute("SELECT AVG(final_score) FROM sessions")
            avg_score = cursor.fetchone()[0] or 0.0

            self.total_hours_label.configure(text=f"Total Hours Focused\n{hours:.1f}h")
            self.total_distractions_label.configure(text=f"Total Distractions\n{total_distractions}")
            self.avg_score_label.configure(text=f"All-Time Avg Score\n{avg_score:.1f}%")
        except Exception as ex:
            print(f"Error updating stats: {ex}")

    def refresh_plot(self):
        """Plot distraction points per session over time."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT date, distraction_points FROM sessions ORDER BY date")
        rows = cursor.fetchall()
        dates = []
        distraction_data = []
        for d, dp in rows:
            try:
                dates.append(datetime.fromisoformat(d))
                distraction_data.append(dp)
            except Exception:
                continue
        
        self.ax.clear()
        self.ax.set_facecolor('#1E1E1E')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.tick_params(colors='white', rotation=45)
        
        if dates:
            self.ax.bar(dates, distraction_data, color='#FF6B6B', alpha=0.7, width=0.3)
        
        self.ax.set_title('Distraction Points per Session', color='white')
        self.ax.set_xlabel('Date', color='white')
        self.ax.set_ylabel('Distraction Points', color='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.fig.autofmt_xdate()
        self.canvas.draw_idle()

    def update_gui(self):
        """Poll queue and update GUI with latest telemetry."""
        # ensure backend thresholds reflect slider positions
        self.backend.yaw_threshold = self.yaw_slider.get()
        self.backend.pitch_threshold = self.pitch_slider.get()
        try:
            data = self.backend.queue.get_nowait()
            
            # update labels
            self.yaw_label.configure(text=f"Yaw: {data['yaw']}°")
            self.pitch_label.configure(text=f"Pitch: {data['pitch']}°")
            self.gr_label.configure(text=f"GR: {data['gr']:.2f}")
            
            # update gamified metrics
            distraction_pts = data.get('distraction_points', 0)
            self.distraction_label.configure(text=f"Distraction Points: {distraction_pts}")
            
            # calculate distraction per minute
            total_time = self.backend.total_session_time
            if total_time > 60.0:  # only show after 60 seconds
                dpm = (distraction_pts / total_time) * 60.0
                self.dpm_label.configure(text=f"Distractions/Min: {dpm:.1f}")
            else:
                self.dpm_label.configure(text="Distractions/Min: 0.0")

            score = data.get('focus_score', 0.0)
            self.score_label.configure(text=f"Score: {score:.1f}%")
            
            # update video frame
            pil_img = cv2_to_pil(data['frame'])
            pil_img.thumbnail((640, 480), Image.Resampling.LANCZOS)
            self.video_tk = ImageTk.PhotoImage(pil_img)
            self.video_frame.configure(image=self.video_tk, text="")
        except queue.Empty:
            pass
        
        # periodically refresh analytics if on that tab (every 10 update cycles = 300ms)
        if not hasattr(self, '_analytics_refresh_count'):
            self._analytics_refresh_count = 0
        self._analytics_refresh_count += 1
        if self._analytics_refresh_count >= 10:
            if self.tabview.get() == "Analytics":
                self.refresh_stats()
                self.refresh_plot()
            self._analytics_refresh_count = 0
        
        self.after(30, self.update_gui)

    def show_tooltip(self, title, description):
        """Show tooltip info."""
        self.info_label.configure(text=f"{title}:\n{description}")
    
    def hide_tooltip(self):
        """Hide tooltip info."""
        self.info_label.configure(text="")
    
    def toggle_pause(self):
        """Toggle pause state."""
        if self.backend.pause_event.is_set():
            self.backend.pause_event.clear()
            self.pause_button.configure(text="Pause")
        else:
            self.backend.pause_event.set()
            self.pause_button.configure(text="Resume")
    
    def quit_app(self):
        """Quit the application and persist session data."""
        self.backend.quit_event.set()
        # store session metrics
        total = self.backend.total_session_time
        focused = self.backend.cumulative_focused_time
        final_score = (focused / total * 100) if total > 0 else 0.0
        distraction_pts = self.backend.distraction_points
        date_str = datetime.now().isoformat(timespec='seconds')
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (date, total_duration, focused_duration, final_score, distraction_points) VALUES (?,?,?,?,?)",
                (date_str, total, focused, final_score, distraction_pts)
            )
            self.conn.commit()
        except Exception as ex:
            print(f"Error saving session to database: {ex}")
        finally:
            try:
                self.conn.close()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = FocusMonitorApp()
    app.mainloop()




