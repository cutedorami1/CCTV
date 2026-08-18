import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

class CCTVController:
    def __init__(self, root):
        self.root = root
        self.root.title("EE IP CCTV Monitor & Signal Analyzer")
        self.root.geometry("1280x720")
        try:
            self.root.state('zoomed')
        except Exception:
            pass

        # 카메라 소스 (기본값: 로컬 웹캠 0)
        self.cap = None
        self.camera_source = "0"

        # YOLO AI 모델 로드
        self.yolo_model = YOLO("yolov8n.pt")

        # 제어 변수
        self.zoom = 1.0
        self.pan = 0
        self.tilt = 0
        self.filter_mode = tk.StringVar(value="Normal")

        self._build_gui()
        self._connect_camera()
        self._update_frame()

    def _build_gui(self):
        # 1. 상단 IP 카메라 입력 및 연결 제어 패널
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(top_frame, text="Camera Source (0 for Webcam or RTSP URL):").pack(side=tk.LEFT, padx=5)
        self.ent_source = ttk.Entry(top_frame, width=50)
        self.ent_source.insert(0, "0")  # 예시: rtsp://admin:1234@192.168.0.100:554/stream1
        self.ent_source.pack(side=tk.LEFT, padx=5)

        btn_connect = ttk.Button(top_frame, text="Connect / Reconnect", command=self._on_connect_click)
        btn_connect.pack(side=tk.LEFT, padx=5)

        # 2. 하단 PTZ & 필터 제어 패널
        ctrl_frame = ttk.LabelFrame(self.root, text=" CCTV PTZ & Filter Controls ")
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        ttk.Label(ctrl_frame, text="Zoom").grid(row=0, column=0, padx=5, pady=5)
        ttk.Scale(ctrl_frame, from_=1.0, to=3.0, value=1.0, command=self._set_zoom).grid(row=0, column=1, sticky="ew")

        ttk.Label(ctrl_frame, text="Pan(X)").grid(row=0, column=2, padx=5, pady=5)
        ttk.Scale(ctrl_frame, from_=-100, to=100, value=0, command=self._set_pan).grid(row=0, column=3, sticky="ew")

        ttk.Label(ctrl_frame, text="Tilt(Y)").grid(row=0, column=4, padx=5, pady=5)
        ttk.Scale(ctrl_frame, from_=-100, to=100, value=0, command=self._set_tilt).grid(row=0, column=5, sticky="ew")

        ttk.Label(ctrl_frame, text="Filter:").grid(row=0, column=6, padx=5, pady=5)
        filters = ["Normal", "Grayscale", "Canny Edge", "YOLO"]
        cb = ttk.Combobox(ctrl_frame, textvariable=self.filter_mode, values=filters, state="readonly", width=12)
        cb.grid(row=0, column=7, padx=5, pady=5)

        for col in [1, 3, 5]:
            ctrl_frame.columnconfigure(col, weight=1)

        # 3. 디스플레이 영역
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.lbl_graph = ttk.Label(display_frame)
        self.lbl_graph.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        self.lbl_video = ttk.Label(display_frame)
        self.lbl_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _connect_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        src_str = self.ent_source.get().strip()
        # 입력값이 숫자인 경우 로컬 장치 번호로, 문자열인 경우 IP/RTSP URL로 변환
        src = int(src_str) if src_str.isdigit() else src_str
        self.cap = cv2.VideoCapture(src)

    def _on_connect_click(self):
        self._connect_camera()

    def _set_zoom(self, val): self.zoom = float(val)
    def _set_pan(self, val): self.pan = int(float(val))
    def _set_tilt(self, val): self.tilt = int(float(val))

    def _process_ptz(self, frame, target_w, target_h):
        h, w, _ = frame.shape
        crop_w, crop_h = int(w / self.zoom), int(h / self.zoom)
        cx, cy = w // 2 + self.pan, h // 2 + self.tilt

        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        
        cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _draw_histogram(self, frame, target_h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        graph_w = 220
        graph_h = max(target_h, 100)
        cv2.normalize(hist, hist, 0, graph_h - 40, cv2.NORM_MINMAX)

        hist_img = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
        for i in range(1, 256):
            x1 = int((i - 1) * ((graph_w - 20) / 256) + 10)
            y1 = graph_h - 20 - int(hist[i - 1][0])
            x2 = int(i * ((graph_w - 20) / 256) + 10)
            y2 = graph_h - 20 - int(hist[i][0])
            cv2.line(hist_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(hist_img, "Brightness Hist", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return hist_img

    def _update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                v_w = self.lbl_video.winfo_width()
                v_h = self.lbl_video.winfo_height()

                target_w = v_w if v_w > 10 else 960
                target_h = v_h if v_h > 10 else 720

                ptz_frame = self._process_ptz(frame, target_w, target_h)

                mode = self.filter_mode.get()
                if mode == "Grayscale":
                    processed = cv2.cvtColor(ptz_frame, cv2.COLOR_BGR2GRAY)
                    display_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                elif mode == "Canny Edge":
                    processed = cv2.Canny(ptz_frame, 100, 200)
                    display_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                elif mode == "YOLO":
                    results = self.yolo_model(ptz_frame, verbose=False)
                    display_img = results[0].plot()
                else:
                    display_img = ptz_frame

                hist_img = self._draw_histogram(display_img, target_h)

                img_v = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)))
                img_g = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)))

                self.lbl_video.config(image=img_v)
                self.lbl_video.image = img_v
                self.lbl_graph.config(image=img_g)
                self.lbl_graph.image = img_g

        self.root.after(30, self._update_frame)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = CCTVController(root)
    root.mainloop()
