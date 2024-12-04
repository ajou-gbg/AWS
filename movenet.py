import numpy as np
from pathlib import Path
import cv2
import argparse
import os
from openvino.runtime import Core
from collections import namedtuple

# 모델 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "./models/movenet_multipose_lightning_256x256_FP32.xml"

# 키포인트와 라인 정의
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

TRACK_COLORS = [(230, 25, 75),
                (60, 180, 75),
                (255, 225, 25),
                (0, 130, 200),
                (245, 130, 48),
                (145, 30, 180),
                (70, 240, 240),
                (240, 50, 230),
                (210, 245, 60),
                (250, 190, 212),
                (0, 128, 128),
                (220, 190, 255),
                (170, 110, 40),
                (255, 250, 200),
                (128, 0, 0),
                (170, 255, 195),
                (128, 128, 0),
                (255, 215, 180),
                (0, 0, 128),
                (128, 128, 128)]

LINES_BODY = [[4, 2], [2, 0], [0, 1], [1, 3], [10, 8], [8, 6], [6, 5], [5, 7], [7, 9],
              [6, 12], [12, 11], [11, 5], [12, 14], [14, 16], [11, 13], [13, 15]]

Padding = namedtuple('Padding', ['w', 'h', 'padded_w', 'padded_h'])


class Body:
    def __init__(self, score, xmin, ymin, xmax, ymax, keypoints_score, keypoints, keypoints_norm):
        self.score = score  # global score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoints_score = keypoints_score  # scores of the keypoints
        self.keypoints_norm = keypoints_norm  # normalized ([0,1]) coordinates
        self.keypoints = keypoints  # keypoints in pixels in the input image

    def print(self):
        attrs = vars(self)
        print('\n'.join(f"{key}: {value}" for key, value in attrs.items()))

    def str_bbox(self):
        return f"xmin={self.xmin} xmax={self.xmax} ymin={self.ymin} ymax={self.ymax}"

class MovenetMPOpenvino:
    def __init__(self, model_path=None, device="CPU", score_thresh=0.11, output=None):
        """
        model_path: Path to the model file.
        device: Device to run the model (e.g., "CPU").
        score_thresh: Minimum confidence score for keypoints.
        output: Path to save the output video (optional).
        """
        self.model_path = DEFAULT_MODEL
        self.device = device
        self.score_thresh = score_thresh
        self.output_path = output
        self.ie = Core()

        # 모델 로드
        self.load_model()

        # 패딩 초기값 (첫 번째 프레임 처리 시 설정)
        self.padding = None

        # 출력 설정
        self.init_output()

    def load_model(self):
        """모델 로드"""
        # print("Loading OpenVINO model...")
        self.model = self.ie.read_model(self.model_path)
        self.compiled_model = self.ie.compile_model(model=self.model, device_name=self.device)

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        input_shape = self.input_layer.shape
        self.pd_h, self.pd_w = input_shape[2], input_shape[3]

    def define_padding(self, frame_shape):
        """패딩 설정 (단일 프레임 기준)"""
        frame_height, frame_width = frame_shape

        model_aspect_ratio = self.pd_w / self.pd_h
        frame_aspect_ratio = frame_width / frame_height

        if frame_aspect_ratio > model_aspect_ratio:
            pad_h = int((frame_width / model_aspect_ratio) - frame_height)
            self.padding = Padding(0, pad_h, frame_width, frame_height + pad_h)
        else:
            pad_w = int((frame_height * model_aspect_ratio) - frame_width)
            self.padding = Padding(pad_w, 0, frame_width + pad_w, frame_height)
        # print(f"Padding: {self.padding}")


    def init_output(self):
        """출력 설정"""
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.output = cv2.VideoWriter(self.output_path, fourcc, 30, (640, 480))  # 기본 해상도 (나중에 업데이트 가능)
        else:
            self.output = None

    def preprocess_frame(self, frame):
        """이미지 전처리"""
        if self.padding is None:
            self.define_padding(frame.shape[:2])  # 첫 번째 프레임에서 패딩 계산

        padded = cv2.copyMakeBorder(frame, 0, self.padding.h, 0, self.padding.w, cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(padded, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
        return resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    
    def render_keypoints(self, frame, bodies):
        """키포인트 렌더링"""
        for body in bodies:
            keypoints = body.keypoints.astype(int)  # 정수형으로 변환
            keypoints_score = body.keypoints_score

            # 키포인트를 연결하는 선 그리기
            for line in LINES_BODY:
                # 선을 이을 두 점의 인덱스 확인
                if keypoints_score[line[0]] > self.score_thresh and keypoints_score[line[1]] > self.score_thresh:
                    pt1 = tuple(keypoints[line[0], :2])  # (x, y)
                    pt2 = tuple(keypoints[line[1], :2])  # (x, y)
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # 초록색 선

            # 각 키포인트에 점 그리기
            for i, kp in enumerate(keypoints):
                if keypoints_score[i] > self.score_thresh:
                    center = tuple(kp[:2])  # (x, y)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)  # 빨간색 점


    def save_frame(self, frame, frame_index):
        """프레임을 파일로 저장"""
        output_file = os.path.join("./received_images", f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(output_file, frame)

    def postprocess(self, inference_result, frame_shape):
        """추론 결과 후처리 및 좌표 복원"""
        result = np.squeeze(inference_result[self.output_layer])  # (6, 56)
        bodies = []

        for person_idx in range(result.shape[0]):
            person_data = result[person_idx]
            score = person_data[55]  # 각 사람의 신뢰도 점수
            # if score > self.score_thresh:
            kps = person_data[:51].reshape(17, 3)
            bbox = person_data[51:55].reshape(2, 2)
            ymin, xmin, ymax, xmax = (bbox * [self.padding.padded_h, self.padding.padded_w]).flatten().astype(int)

            kps[:, 0] = kps[:, 0] * self.padding.padded_w
            kps[:, 1] = kps[:, 1] * self.padding.padded_h

            # 키포인트 배열을 (x, y, confidence)로 설정
            # keypoints_with_confidence = np.hstack([kps[:, :2], kps[:, 2:3]])
            kps[:, [0, 1]] = kps[:, [1, 0]]

            body = Body(
                score=score,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                keypoints_score=kps[:, 2],
                keypoints=kps.astype(float),
                keypoints_norm=kps[:, [1, 0]] / np.array([frame_shape[1], frame_shape[0]])
            )
            bodies.append(body)

        return bodies


    def run(self, frames):
        """
        frames: List of input frames (images) or paths to image files.
        """
        processed_frames = []

        for i, frame in enumerate(frames):
            # 파일 경로일 경우 이미지를 로드
            if isinstance(frame, str):
                frame = cv2.imread(frame)
                if frame is None:
                    print(f"Error loading image: {frame}")
                    continue

            # 전처리 및 추론
            preprocessed = self.preprocess_frame(frame)
            inference_result = self.compiled_model([preprocessed])
            bodies = self.postprocess(inference_result, frame.shape)
            self.render_keypoints(frame, bodies)
            
            self.save_frame(frame, i)

            # 추출된 keypoints (6, 17, 3) 형태로 정리
            frame_keypoints = np.zeros((6, 17, 3), dtype=np.float32)  # 6명의 사람, 17개 관절, (x, y, confidence)
            for i, body in enumerate(bodies):
                frame_keypoints[i, :, :] = body.keypoints  # 각 Body의 키포인트 좌표 추가
            processed_frames.append(frame_keypoints)

        return np.array(processed_frames, dtype=np.float32)  # (num_frames, 6, 17, 3)



