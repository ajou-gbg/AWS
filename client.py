import requests
import cv2
import numpy as np

# 서버 URL
VIDEO_FEED_URL = "http://203.253.70.226:5000/video_feed"

def stream_video():
    try:
        # 서버로부터 스트리밍 데이터 요청
        response = requests.get(VIDEO_FEED_URL, stream=True)
        if response.status_code == 200:
            print("Connected to video feed.")
            byte_data = b""
            for chunk in response.iter_content(chunk_size=1024):
                byte_data += chunk
                # 프레임의 시작과 끝을 찾습니다.
                a = byte_data.find(b'\xff\xd8')  # JPEG 시작
                b = byte_data.find(b'\xff\xd9')  # JPEG 끝
                if a != -1 and b != -1:
                    # 프레임 데이터를 추출합니다.
                    jpg_data = byte_data[a:b + 2]
                    byte_data = byte_data[b + 2:]

                    # 프레임을 디코딩하여 OpenCV로 처리
                    frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
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
