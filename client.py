import requests
import cv2
import numpy as np
import time
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 서버 URL
VIDEO_FEED_URL = os.getenv("VIDEO_FEED_URL")  # 비디오 피드 URL

def stream_video():
    try:
        # 서버로부터 스트리밍 데이터 요청
        response = requests.get(VIDEO_FEED_URL, stream=True)
        if response.status_code == 200:
            print("Connected to video feed.")
            byte_data = b""
            last_frame_time = 0
            frame_interval = 1 / 30  # 30 FPS 기준

            for chunk in response.iter_content(chunk_size=1024):
                byte_data += chunk

                # 프레임의 시작과 끝을 찾습니다.
                a = byte_data.find(b'\xff\xd8')  # JPEG 시작
                b = byte_data.find(b'\xff\xd9')  # JPEG 끝
                if a != -1 and b != -1:
                    # 프레임 데이터를 추출합니다.
                    jpg_data = byte_data[a:b + 2]
                    byte_data = byte_data[b + 2:]  # 사용한 데이터는 버퍼에서 제거

                    # 현재 시간과 이전 프레임의 시간 비교
                    current_time = time.time()
                    if current_time - last_frame_time < frame_interval:
                        continue  # 프레임 속도를 조절하여 과도한 렌더링 방지
                    last_frame_time = current_time

                    # 프레임을 디코딩하여 OpenCV로 처리
                    frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        # 프레임 크기 조정 (해상도 줄이기)
                        frame = cv2.resize(frame, (640, 480))  # 해상도: 640x480

                        # 프레임을 화면에 표시
                        cv2.imshow("Client Video Feed", frame)

                        # 'q' 키로 종료
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        else:
            print(f"Failed to connect: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
