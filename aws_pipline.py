import os
import socket
import cv2
import numpy as np
import struct
import time
from flask import Flask, Response
from collections import deque
from movenet import MovenetMPOpenvino
from tst import TSTModel
import torch

# Flask application initialization
app = Flask(__name__)

sliding_window = deque(maxlen=15)  # 최대 15개의 프레임 저장
send_frames = []

# Initialize MoveNet and TSTModel
input_dim = 6 * 17 * 3  # MoveNet keypoints 차원: 6명의 사람 * 17개의 keypoints * (x, y, confidence)
num_classes = 3
movenet = MovenetMPOpenvino()
tst = TSTModel(input_dim=input_dim, num_classes=num_classes).to('cpu')  # 모델을 CPU에서 실행
tst.load_model("tst.pth")


# Function to process frames in the sliding window
def process_sliding_window(frames):
    # MoveNet을 사용하여 키포인트 처리
    keypoints_batch = process_frame(frames)  # (15, 6*17*3)

    # TSTModel 입력 준비
    tst_input = torch.tensor(keypoints_batch, dtype=torch.float32).unsqueeze(0)  # (1, 15, 6*17*3)

    # TSTModel로 예측
    results = tst.inference(tst_input)

    # 가장 가능성 높은 클래스를 추출

    # 프레임에 예측 결과 삽입
    for frame in frames:
        text = f"Class: {results}"
        cv2.putText(
            frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

    return frames

# Generate frames for real-time streaming
def generate_frames():
    global send_frames
    while True:
        if send_frames:
            latest_frames = send_frames[-5:]  # 최신 프레임 최대 5개 가져오기
            frames_encoded = []

            for frame in latest_frames:
                if frame is not None and isinstance(frame, np.ndarray):
                    # 프레임을 JPEG로 인코딩
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames_encoded.append(buffer.tobytes())
                else:
                    # 비어있거나 잘못된 프레임 처리
                    frames_encoded.append(None)

            for frame_bytes in frames_encoded:
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)  # 슬라이딩 윈도우가 비어 있으면 잠시 대기


# Flask route for real-time video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def process_frame(frames):
    keypoints = movenet.run(frames)  # (6, 17, 3) 형태의 keypoints
    # x, y, confidence를 포함하여 (1, 6*17*3) 형태로 변환
    num_frames = keypoints.shape[0]
    keypoints_flattened = keypoints.reshape(num_frames, -1)
    return keypoints_flattened


# Socket server function
def socket_server():
    global send_frames
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 5002))  # Listening on port 5002
    server_socket.listen(1)
    print("Socket server is listening on port 5002...")

    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")
    buffer = []
    while True:
        time.sleep(0.01)
        try:
            # Receive data size
            packed_size = client_socket.recv(struct.calcsize('>Q'))
            if not packed_size:
                print("No data received for frame size. Continuing...")
                continue

            # Unpack data size
            try:
                msg_size = struct.unpack('>Q', packed_size)[0]
                # print(f"Expecting frame of size: {msg_size} bytes")
            except struct.error as e:
                print(f"Failed to unpack frame size: {e}")
                continue

            # Receive data
            data = b""
            while len(data) < msg_size:
                remaining = msg_size - len(data)
                packet = client_socket.recv(4096 if remaining > 4096 else remaining)
                if not packet:
                    print("Received incomplete packet. Ignoring remaining bytes...")
                    break
                data += packet

            if len(data) != msg_size:
                print(f"Received incomplete frame. Expected {msg_size}, got {len(data)}. Continuing...")
                continue

            # Decode the received image
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if frame is not None:
                buffer.append(frame)  # Add frame to the buffer
                print("Hi")
                if len(buffer) == 5:
                    if len(sliding_window) == 15:
                        send_frames = process_sliding_window(list(sliding_window))  # Process the sliding window
                        # Remove oldest frames to maintain max length of 15
                        for _ in range(len(buffer)):
                            sliding_window.popleft()

                    sliding_window.extend(buffer)  # Add 5 frames to the sliding window
                    buffer.clear()  # Clear the buffer
            else:
                print("Failed to decode frame. Continuing...")

        except Exception as e:
            print(f"Error during frame handling: {e}")
            continue

    client_socket.close()
    server_socket.close()


# Run both socket server and Flask server
if __name__ == "__main__":
    import threading

    # Start the socket server thread
    socket_thread = threading.Thread(target=socket_server, daemon=True)
    socket_thread.start()

    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
    