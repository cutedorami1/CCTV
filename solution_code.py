# OpenCV 라이브러리
# 웹캠/IP 카메라 영상 입력, 영상 처리, 색상 변환 등에 사용
import cv2

# NumPy 라이브러리
# 배열 및 히스토그램 이미지를 만들 때 사용
import numpy as np

# Python 기본 GUI 라이브러리 Tkinter
import tkinter as tk

# Tkinter의 조금 더 현대적인 디자인의 위젯 사용
from tkinter import ttk

# OpenCV 이미지를 Tkinter에서 표시할 수 있도록 변환하기 위해 사용
from PIL import Image, ImageTk

# YOLO 객체 탐지 모델을 사용하기 위한 라이브러리
from ultralytics import YOLO


# ============================================================
# CCTV 제어 프로그램의 전체 기능을 담당하는 클래스
# ============================================================
class CCTVController:

    # --------------------------------------------------------
    # 클래스가 생성될 때 처음 실행되는 초기화 함수
    # --------------------------------------------------------
    def __init__(self, root):

        # Tkinter의 메인 윈도우를 저장
        self.root = root

        # 프로그램 창 제목 설정
        self.root.title("EE IP CCTV Monitor & Signal Analyzer")

        # 기본 프로그램 창 크기 설정
        # 가로 1280px, 세로 720px
        self.root.geometry("1280x720")

        # 프로그램 실행 시 창을 최대화
        try:
            self.root.state('zoomed')

        # 일부 운영체제에서는 zoomed가 지원되지 않을 수 있으므로
        # 오류가 발생해도 프로그램이 종료되지 않도록 처리
        except Exception:
            pass


        # ====================================================
        # 카메라 관련 변수
        # ====================================================

        # OpenCV의 VideoCapture 객체를 저장할 변수
        # 아직 카메라를 연결하지 않았으므로 None으로 설정
        self.cap = None

        # 기본 카메라 번호
        # "0"은 일반적으로 컴퓨터의 기본 웹캠을 의미
        self.camera_source = "0"


        # ====================================================
        # YOLO AI 모델
        # ====================================================

        # YOLOv8 Nano 모델을 불러옴
        # 사람, 자동차, 동물 등의 객체를 실시간으로 탐지할 수 있음
        self.yolo_model = YOLO("yolov8n.pt")


        # ====================================================
        # PTZ 및 필터 제어 변수
        # ====================================================

        # 디지털 Zoom 배율
        # 1.0 = 확대 없음
        self.zoom = 1.0

        # 영상의 좌우 이동 값
        self.pan = 0

        # 영상의 상하 이동 값
        self.tilt = 0

        # 현재 선택된 영상 처리 모드
        # 기본값은 Normal
        self.filter_mode = tk.StringVar(value="Normal")


        # ====================================================
        # GUI 생성
        # ====================================================

        # 프로그램의 GUI 요소 생성
        self._build_gui()

        # 카메라 연결
        self._connect_camera()

        # 영상 업데이트 시작
        self._update_frame()


    # ========================================================
    # GUI를 만드는 함수
    # ========================================================
    def _build_gui(self):

        # ====================================================
        # 1. 상단 카메라 연결 패널
        # ====================================================

        # 상단 프레임 생성
        top_frame = ttk.Frame(self.root)

        # 프레임을 프로그램 상단에 배치
        # fill=tk.X → 가로 방향으로 창 전체를 채움
        top_frame.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=5
        )


        # 카메라 입력 주소 설명 Label
        ttk.Label(
            top_frame,
            text="Camera Source (0 for Webcam or RTSP URL):"
        ).pack(
            side=tk.LEFT,
            padx=5
        )


        # 카메라 주소를 입력하는 Entry 생성
        self.ent_source = ttk.Entry(
            top_frame,
            width=50
        )

        # 기본값으로 "0" 입력
        # 0은 PC의 기본 웹캠
        self.ent_source.insert(0, "0")

        # 입력창을 왼쪽부터 배치
        self.ent_source.pack(
            side=tk.LEFT,
            padx=5
        )


        # 카메라 연결/재연결 버튼
        btn_connect = ttk.Button(
            top_frame,

            # 버튼에 표시되는 문자
            text="Connect / Reconnect",

            # 버튼을 누르면 실행될 함수
            command=self._on_connect_click
        )

        # 버튼 배치
        btn_connect.pack(
            side=tk.LEFT,
            padx=5
        )


        # ====================================================
        # 2. 하단 PTZ 및 필터 제어 패널
        # ====================================================

        # 테두리와 제목이 있는 LabelFrame 생성
        ctrl_frame = ttk.LabelFrame(
            self.root,
            text=" CCTV PTZ & Filter Controls "
        )

        # 프로그램 하단에 배치
        ctrl_frame.pack(
            side=tk.BOTTOM,
            fill=tk.X,
            padx=10,
            pady=5
        )


        # ----------------------------------------------------
        # Zoom 제어
        # ----------------------------------------------------

        # Zoom 문자 표시
        ttk.Label(
            ctrl_frame,
            text="Zoom"
        ).grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )


        # Zoom 슬라이더
        ttk.Scale(
            ctrl_frame,

            # 최소 확대 배율
            from_=1.0,

            # 최대 확대 배율
            to=3.0,

            # 기본값
            value=1.0,

            # 슬라이더가 움직이면 _set_zoom 함수 호출
            command=self._set_zoom

        ).grid(
            row=0,
            column=1,
            sticky="ew"
        )


        # ----------------------------------------------------
        # Pan 제어
        # 영상의 좌우 위치 이동
        # ----------------------------------------------------

        ttk.Label(
            ctrl_frame,
            text="Pan(X)"
        ).grid(
            row=0,
            column=2,
            padx=5,
            pady=5
        )


        # Pan 슬라이더
        ttk.Scale(
            ctrl_frame,

            # 왼쪽 최대 이동
            from_=-100,

            # 오른쪽 최대 이동
            to=100,

            # 기본 위치
            value=0,

            # 슬라이더 변경 시 호출
            command=self._set_pan

        ).grid(
            row=0,
            column=3,
            sticky="ew"
        )


        # ----------------------------------------------------
        # Tilt 제어
        # 영상의 상하 위치 이동
        # ----------------------------------------------------

        ttk.Label(
            ctrl_frame,
            text="Tilt(Y)"
        ).grid(
            row=0,
            column=4,
            padx=5,
            pady=5
        )


        # Tilt 슬라이더
        ttk.Scale(
            ctrl_frame,

            # 위쪽/아래쪽 이동 범위
            from_=-100,
            to=100,

            # 기본값
            value=0,

            # 슬라이더 변경 시 호출
            command=self._set_tilt

        ).grid(
            row=0,
            column=5,
            sticky="ew"
        )


        # ----------------------------------------------------
        # 영상 필터 선택
        # ----------------------------------------------------

        ttk.Label(
            ctrl_frame,
            text="Filter:"
        ).grid(
            row=0,
            column=6,
            padx=5,
            pady=5
        )


        # 사용할 수 있는 영상 처리 방식
        filters = [
            "Normal",       # 원본 영상
            "Grayscale",    # 흑백 영상
            "Canny Edge",   # 윤곽선 검출
            "YOLO"          # AI 객체 탐지
        ]


        # 필터를 선택할 수 있는 콤보박스 생성
        cb = ttk.Combobox(
            ctrl_frame,

            # 선택 결과를 filter_mode 변수에 저장
            textvariable=self.filter_mode,

            # 선택 가능한 값
            values=filters,

            # 사용자가 직접 입력하지 못하고
            # 목록에서만 선택하도록 설정
            state="readonly",

            # 콤보박스 폭
            width=12
        )

        # 콤보박스 위치
        cb.grid(
            row=0,
            column=7,
            padx=5,
            pady=5
        )


        # Zoom, Pan, Tilt 슬라이더 영역이
        # 창 크기에 따라 늘어나도록 설정
        for col in [1, 3, 5]:
            ctrl_frame.columnconfigure(
                col,
                weight=1
            )


        # ====================================================
        # 3. CCTV 영상 및 그래프 표시 영역
        # ====================================================

        # 영상 표시용 Frame
        display_frame = ttk.Frame(self.root)

        # 남은 화면 전체를 차지하도록 설정
        display_frame.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=5
        )


        # ----------------------------------------------------
        # 오른쪽 히스토그램 영역
        # ----------------------------------------------------

        self.lbl_graph = ttk.Label(display_frame)

        # 오른쪽에 배치
        self.lbl_graph.pack(
            side=tk.RIGHT,
            fill=tk.Y,
            padx=(5, 0)
        )


        # ----------------------------------------------------
        # 왼쪽 CCTV 영상 영역
        # ----------------------------------------------------

        self.lbl_video = ttk.Label(display_frame)

        # 남은 공간 전체를 영상이 사용
        self.lbl_video.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )


    # ========================================================
    # 카메라 연결 함수
    # ========================================================
    def _connect_camera(self):

        # 기존에 연결된 카메라가 있다면
        if self.cap is not None and self.cap.isOpened():

            # 기존 카메라 연결 해제
            self.cap.release()


        # 입력창에 작성된 내용을 가져옴
        # strip()은 앞뒤 공백 제거
        src_str = self.ent_source.get().strip()


        # 입력값이 숫자인지 확인
        #
        # 예:
        # "0" → 숫자 → 웹캠 번호 0
        # "1" → 숫자 → 웹캠 번호 1
        #
        # "rtsp://..." → 문자열 → 네트워크 카메라 주소
        src = int(src_str) if src_str.isdigit() else src_str


        # OpenCV를 이용하여 카메라 연결
        self.cap = cv2.VideoCapture(src)


    # ========================================================
    # Connect / Reconnect 버튼을 눌렀을 때 실행
    # ========================================================
    def _on_connect_click(self):

        # 카메라 연결 함수 실행
        self._connect_camera()


    # ========================================================
    # Zoom 슬라이더 값 저장
    # ========================================================
    def _set_zoom(self, val):

        # Tkinter Scale에서 전달된 값을
        # 실수(float)로 변환하여 저장
        self.zoom = float(val)


    # ========================================================
    # Pan 슬라이더 값 저장
    # ========================================================
    def _set_pan(self, val):

        # 값을 정수로 변환하여 저장
        self.pan = int(float(val))


    # ========================================================
    # Tilt 슬라이더 값 저장
    # ========================================================
    def _set_tilt(self, val):

        # 값을 정수로 변환하여 저장
        self.tilt = int(float(val))


    # ========================================================
    # 디지털 PTZ 처리 함수
    #
    # 실제 카메라 모터를 움직이는 것이 아니라
    # 영상의 일부를 잘라내고 확대하는 방식
    # ========================================================
    def _process_ptz(self, frame, target_w, target_h):

        # 현재 영상의 높이(h), 너비(w), 채널 수를 가져옴
        h, w, _ = frame.shape


        # Zoom 값에 따라 잘라낼 영상 크기를 계산
        #
        # zoom = 1 → 원본 크기
        # zoom = 2 → 원본의 절반 크기를 잘라낸 뒤 확대
        # zoom = 3 → 원본의 1/3 크기를 잘라낸 뒤 확대
        crop_w = int(w / self.zoom)
        crop_h = int(h / self.zoom)


        # 영상의 기본 중심 좌표
        #
        # Pan 값으로 X축 이동
        # Tilt 값으로 Y축 이동
        cx = w // 2 + self.pan
        cy = h // 2 + self.tilt


        # 잘라낼 영역의 왼쪽 위 X 좌표 계산
        #
        # max/min을 사용하여 영상 범위를 벗어나지 않도록 제한
        x1 = max(
            0,
            min(
                w - crop_w,
                cx - crop_w // 2
            )
        )


        # 잘라낼 영역의 왼쪽 위 Y 좌표 계산
        y1 = max(
            0,
            min(
                h - crop_h,
                cy - crop_h // 2
            )
        )


        # 계산된 영역만 잘라냄
        cropped = frame[
            y1:y1 + crop_h,
            x1:x1 + crop_w
        ]


        # 잘라낸 영상을 GUI 크기에 맞게 다시 확대
        return cv2.resize(
            cropped,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )


    # ========================================================
    # 영상 밝기 히스토그램 생성 함수
    # ========================================================
    def _draw_histogram(self, frame, target_h):

        # 영상이 컬러 영상이면
        if len(frame.shape) == 3:

            # BGR 컬러 영상을 흑백으로 변환
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

        # 이미 흑백 영상이라면 그대로 사용
        else:
            gray = frame


        # 흑백 영상의 밝기 히스토그램 계산
        #
        # 밝기 값:
        # 0   = 검정
        # 255 = 흰색
        hist = cv2.calcHist(
            [gray],         # 분석할 영상
            [0],            # 채널
            None,           # 마스크 없음
            [256],          # 256단계 밝기
            [0, 256]        # 밝기 범위
        )


        # 그래프 가로 크기
        graph_w = 220

        # 그래프 세로 크기
        # 최소 높이는 100px
        graph_h = max(target_h, 100)


        # 히스토그램 값을 화면 높이에 맞게 정규화
        cv2.normalize(
            hist,
            hist,
            0,
            graph_h - 40,
            cv2.NORM_MINMAX
        )


        # 검은색 그래프 배경 이미지 생성
        #
        # graph_h = 높이
        # graph_w = 너비
        # 3 = BGR 컬러 채널
        hist_img = np.zeros(
            (graph_h, graph_w, 3),
            dtype=np.uint8
        )


        # 1 ~ 255 밝기 값에 대해 반복
        for i in range(1, 256):

            # 이전 밝기 값의 X 좌표
            x1 = int(
                (i - 1) * ((graph_w - 20) / 256) + 10
            )

            # 이전 밝기 값의 Y 좌표
            y1 = graph_h - 20 - int(hist[i - 1][0])


            # 현재 밝기 값의 X 좌표
            x2 = int(
                i * ((graph_w - 20) / 256) + 10
            )

            # 현재 밝기 값의 Y 좌표
            y2 = graph_h - 20 - int(hist[i][0])


            # 이전 점과 현재 점을 선으로 연결
            cv2.line(
                hist_img,
                (x1, y1),
                (x2, y2),

                # 선 색상: 초록색(BGR)
                (0, 255, 0),

                # 선 두께
                1
            )


        # 그래프 상단에 제목 출력
        cv2.putText(
            hist_img,

            # 출력할 문자
            "Brightness Hist",

            # 글자 위치
            (10, 25),

            # 폰트
            cv2.FONT_HERSHEY_SIMPLEX,

            # 글자 크기
            0.45,

            # 글자 색상: 흰색
            (255, 255, 255),

            # 글자 두께
            1
        )


        # 완성된 히스토그램 이미지 반환
        return hist_img


    # ========================================================
    # 카메라 영상을 계속 업데이트하는 핵심 함수
    # ========================================================
    def _update_frame(self):

        # 카메라 객체가 존재하고 정상적으로 열려 있는지 확인
        if self.cap is not None and self.cap.isOpened():

            # 카메라에서 현재 프레임 1장을 읽음
            #
            # ret   → 읽기 성공 여부
            # frame → 실제 카메라 이미지
            ret, frame = self.cap.read()


            # 영상 읽기에 성공했다면
            if ret:

                # 영상을 좌우 반전
                #
                # 1 = 좌우 반전
                # 웹캠을 거울처럼 보이게 함
                frame = cv2.flip(frame, 1)


                # 현재 GUI의 영상 Label 너비
                v_w = self.lbl_video.winfo_width()

                # 현재 GUI의 영상 Label 높이
                v_h = self.lbl_video.winfo_height()


                # GUI가 아직 완전히 생성되지 않아
                # 크기가 너무 작게 측정될 경우 기본값 960 사용
                target_w = v_w if v_w > 10 else 960

                # 높이도 같은 방식으로 기본값 720 사용
                target_h = v_h if v_h > 10 else 720


                # Zoom / Pan / Tilt 처리
                ptz_frame = self._process_ptz(
                    frame,
                    target_w,
                    target_h
                )


                # 현재 사용자가 선택한 필터 확인
                mode = self.filter_mode.get()


                # =================================================
                # Grayscale 모드
                # =================================================
                if mode == "Grayscale":

                    # 컬러 영상을 흑백으로 변환
                    processed = cv2.cvtColor(
                        ptz_frame,
                        cv2.COLOR_BGR2GRAY
                    )

                    # Tkinter에서 처리하기 편하도록
                    # 다시 3채널 BGR 영상으로 변환
                    display_img = cv2.cvtColor(
                        processed,
                        cv2.COLOR_GRAY2BGR
                    )


                # =================================================
                # Canny Edge 모드
                # =================================================
                elif mode == "Canny Edge":

                    # Canny 알고리즘을 이용하여
                    # 영상의 윤곽선 검출
                    processed = cv2.Canny(
                        ptz_frame,

                        # 하위 Threshold
                        100,

                        # 상위 Threshold
                        200
                    )

                    # 흑백 윤곽선 영상을
                    # 3채널 BGR로 변환
                    display_img = cv2.cvtColor(
                        processed,
                        cv2.COLOR_GRAY2BGR
                    )


                # =================================================
                # YOLO 객체 탐지 모드
                # =================================================
                elif mode == "YOLO":

                    # YOLO 모델에 현재 영상을 입력
                    #
                    # verbose=False
                    # → 터미널에 탐지 로그를 계속 출력하지 않음
                    results = self.yolo_model(
                        ptz_frame,
                        verbose=False
                    )


                    # YOLO가 탐지한 결과를 영상 위에 그림
                    #
                    # 예:
                    # person
                    # car
                    # dog
                    # 등의 Bounding Box 표시
                    display_img = results[0].plot()


                # =================================================
                # Normal 모드
                # =================================================
                else:

                    # 별도의 영상 처리 없이
                    # PTZ 처리된 원본 영상 사용
                    display_img = ptz_frame


                # =================================================
                # 밝기 히스토그램 생성
                # =================================================

                hist_img = self._draw_histogram(
                    display_img,
                    target_h
                )


                # =================================================
                # OpenCV → Tkinter 이미지 변환
                # =================================================

                # OpenCV는 BGR 순서 사용
                # PIL은 RGB 순서 사용
                #
                # 따라서 BGR → RGB 변환 후
                # PIL Image로 변환하고
                # Tkinter PhotoImage로 변환
                img_v = ImageTk.PhotoImage(
                    Image.fromarray(
                        cv2.cvtColor(
                            display_img,
                            cv2.COLOR_BGR2RGB
                        )
                    )
                )


                # 히스토그램 이미지도 동일한 방식으로 변환
                img_g = ImageTk.PhotoImage(
                    Image.fromarray(
                        cv2.cvtColor(
                            hist_img,
                            cv2.COLOR_BGR2RGB
                        )
                    )
                )


                # =================================================
                # GUI 화면에 CCTV 영상 표시
                # =================================================

                # Label에 CCTV 이미지 설정
                self.lbl_video.config(
                    image=img_v
                )

                # PhotoImage 객체가 Garbage Collection으로
                # 삭제되는 것을 방지하기 위해 참조 유지
                self.lbl_video.image = img_v


                # =================================================
                # GUI 화면에 히스토그램 표시
                # =================================================

                self.lbl_graph.config(
                    image=img_g
                )

                # 이미지 참조 유지
                self.lbl_graph.image = img_g


        # ====================================================
        # 약 30ms 후 다시 _update_frame 함수 호출
        #
        # 이 동작을 반복하면서 실시간 영상처럼 보이게 됨
        #
        # 이론상:
        # 1000ms / 30ms ≈ 33 FPS
        #
        # 단, YOLO 처리 시간 등에 따라 실제 FPS는 더 낮을 수 있음
        # ====================================================
        self.root.after(
            30,
            self._update_frame
        )


    # ========================================================
    # CCTVController 객체가 삭제될 때 실행
    # ========================================================
    def __del__(self):

        # cap 속성이 존재하고
        # 카메라가 연결되어 있다면
        if (
            hasattr(self, 'cap')
            and self.cap is not None
            and self.cap.isOpened()
        ):

            # 카메라 장치 연결 해제
            self.cap.release()


# ============================================================
# 프로그램 시작 부분
# ============================================================

# 이 Python 파일을 직접 실행했을 때만 아래 코드 실행
if __name__ == "__main__":

    # Tkinter 메인 윈도우 생성
    root = tk.Tk()

    # CCTVController 객체 생성
    #
    # 이 순간 __init__()이 실행되면서
    # GUI 생성 → YOLO 로드 → 카메라 연결 → 영상 출력이 시작됨
    app = CCTVController(root)

    # Tkinter 이벤트 루프 실행
    #
    # 프로그램 창이 닫힐 때까지
    # 버튼, 슬라이더, 화면 갱신 등의 이벤트를 계속 처리
    root.mainloop()
