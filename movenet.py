# import tensorflow as tf
# import tensorflow_hub as hub

# class MoveNet:
#     def __init__(self, model_url='https://tfhub.dev/google/movenet/multipose/lightning/1'):
#         self.model = hub.load(model_url)
#         self.movenet = self.model.signatures['serving_default']

#     def preprocess_image(self, image):
#         # 이미지 차원 확장 및 크기 조정
#         preprocessed_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 352, 640)
#         return tf.cast(preprocessed_image, dtype=tf.int32)

#     def predict_keypoints(self, image):
#         preprocessed_image = self.preprocess_image(image)
#         results = self.movenet(preprocessed_image)
#         keypoints = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
#         return keypoints

# if __name__ == "__main__":
#     import cv2
#     img = cv2.imread('test_image.jpg')
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
#     movenet = MoveNet()
#     keypoints = movenet.predict_keypoints(img_rgb)
#     print(keypoints)

import numpy as np
from pathlib import Path
import cv2
import argparse
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

    def postprocess(self, inference_result, frame_shape):
        """추론 결과 후처리 및 좌표 복원"""
        result = np.squeeze(inference_result[self.output_layer])  # (6, 56)
        bodies = []

        for person_idx in range(result.shape[0]):
            person_data = result[person_idx]
            score = person_data[55]  # 각 사람의 신뢰도 점수
            if score > self.score_thresh:
                kps = person_data[:51].reshape(17, 3)
                bbox = person_data[51:55].reshape(2, 2)
                ymin, xmin, ymax, xmax = (bbox * [self.padding.padded_h, self.padding.padded_w]).flatten().astype(int)

                kps[:, 0] = kps[:, 0] * self.padding.padded_w
                kps[:, 1] = kps[:, 1] * self.padding.padded_h
                kps[:, 0] -= self.padding.w
                kps[:, 1] -= self.padding.h
                kps[:, 0] = np.clip(kps[:, 0], 0, frame_shape[1] - 1)
                kps[:, 1] = np.clip(kps[:, 1], 0, frame_shape[0] - 1)

                # 키포인트 배열을 (x, y, confidence)로 설정
                keypoints_with_confidence = np.hstack([kps[:, :2], kps[:, 2:3]])

                body = Body(
                    score=score,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    keypoints_score=kps[:, 2],
                    keypoints=keypoints_with_confidence.astype(float),
                    keypoints_norm=kps[:, [1, 0]] / np.array([frame_shape[1], frame_shape[0]])
                )
                bodies.append(body)

        return bodies


    def run(self, frames):
        """
        frames: List of input frames (images) or paths to image files.
        """
        processed_frames = []

        for frame in frames:
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

            # 추출된 keypoints (6, 17, 3) 형태로 정리
            frame_keypoints = np.zeros((6, 17, 3), dtype=np.float32)  # 6명의 사람, 17개 관절, (x, y, confidence)
            for i, body in enumerate(bodies):
                frame_keypoints[i, :, :] = body.keypoints  # 각 Body의 키포인트 좌표 추가
            processed_frames.append(frame_keypoints)

        return np.array(processed_frames, dtype=np.float32)  # (num_frames, 6, 17, 3)



