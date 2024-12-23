import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from models.movenet import MovenetMPOpenvino
from models.tst import TSTModel
import time
import torch

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames

def process_frame(frames):
    keypoints = movenet.run(frames)  # (6, 17, 3) 형태의 keypoints
    # x, y, confidence를 포함하여 (1, 6*17*3) 형태로 변환
    num_frames = keypoints.shape[0]
    keypoints_flattened = keypoints.reshape(num_frames, -1)
    return keypoints_flattened

# def process_frames(frames):
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         keypoints_list = list(executor.map(process_frame, frames))
#     return np.array(keypoints_list)  # (num_frames, keypoints_dim)

# def process_frames_serial(frames):
#     keypoints_list = []
#     for frame in frames:
#         keypoints = process_frame(frame)  # 프레임 개별 처리
#         keypoints_list.append(keypoints)
#     return np.array(keypoints_list)  # (num_frames, keypoints_dim)

if __name__ == "__main__":

    # 초기 설정
    input_dim = 6 * 17 * 3  # keypoints 차원: 6명의 사람 * 17개의 keypoints * (x, y, confidence)
    num_classes = 3
    video_path = "assault.mp4"  # 처리할 동영상 파일 경로

    movenet = MovenetMPOpenvino()
    tst = TSTModel(input_dim=input_dim, num_classes=num_classes) 
    tst.load_model("./checkpoints/tst.pth")
            #    load_model_path="tst_model.pth", device='cpu')

    # 동영상에서 모든 프레임 추출
    start_time = time.time()  # 시작 시간 측정
    frames = extract_frames(video_path)
    extract_time = time.time()  # 프레임 추출 완료 시간
    print(f"Time taken to extract frames: {extract_time - start_time:.2f} seconds")

    # MoveNet을 통해 모든 프레임 처리
    process_start_time = time.time()  # MoveNet 처리 시작 시간
    keypoints_batch = process_frame(frames[:15])  # (15, 6*17*3)
    process_time = time.time()  # MoveNet 처리 완료 시간
    print(f"Time taken for MoveNet processing: {process_time - process_start_time:.2f} seconds")
    
    tst_input = torch.tensor(keypoints_batch, dtype=torch.float32).unsqueeze(0)  # (1, 15, 6*17*3)

    # TSTModel으로 예측
    predict_start_time = time.time()  # TSTModel 예측 시작 시간
    results = tst.inference(tst_input)  # (1, num_classes)
    predict_time = time.time()  # 예측 완료 시간
    print(f"Time taken for TSTModel prediction: {predict_time - predict_start_time:.2f} seconds")

    # 전체 시간
    total_time = predict_time - start_time
    print(f"Total time taken from frame extraction to TSTModel prediction: {total_time:.2f} seconds")

    print("Results:", results)
