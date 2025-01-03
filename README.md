# README

## 개요
이 프로젝트는 실시간 영상 스트리밍과 추론을 위한 세 가지 구성 요소로 이루어져 있습니다. 각각의 구성 요소는 다음과 같습니다:

1. **camera (열화상 카메라)**: 열화상 카메라로부터 프레임을 캡처하여 AWS 서버로 전송합니다.
2. **AWS 서버**: 라즈베리파이로부터 수신된 프레임을 처리한 뒤, 클라이언트로 결과를 스트리밍합니다.
3. **클라이언트**: AWS 서버로부터 처리된 프레임을 수신하여 실시간으로 표시합니다.

## 패키지 설치
```bash
pip install -r requirements.txt
```

## 구성 요소

### 1. camera 코드
이 코드는 열화상 카메라에서 프레임을 캡처하고 AWS 서버로 소켓 연결을 통해 전송합니다.

#### 주요 기능:
- 로컬 열화상 카메라 연결.
- 실시간으로 프레임 캡처 및 처리.
- AWS 서버로 프레임 전송.

#### 실행:
```bash
python camera.py
```

---

### 2. AWS 서버 코드
이 코드는 라즈베리파이로부터 프레임을 수신하고, 머신러닝 모델을 사용하여 처리한 뒤 Flask를 통해 클라이언트에 스트리밍합니다.

#### 주요 기능:
- 소켓을 사용하여 라즈베리파이에서 프레임 수신.
- MoveNet 및 TSTModel 파이프라인으로 프레임 처리.
- Flask를 통해 클라이언트에 처리된 프레임 스트리밍.

- 머신러닝 모델:
  - MoveNet: `MovenetMPOpenvino`
  - TSTModel: `TSTModel`

#### 실행:
```bash
python aws.py
```

---

### 3. 클라이언트 코드
이 코드는 AWS Flask 서버에 연결하여 실시간으로 처리된 비디오 프레임을 수신 및 표시합니다.

#### 주요 기능:
- AWS Flask 서버에서 프레임 스트리밍.
- 실시간 비디오 피드 표시.

#### 실행:
```bash
python client.py
```

---

## 아키텍처
1. **라즈베리파이**:
   - 열화상 이미지를 캡처.
   - 소켓 통신을 통해 AWS 서버로 프레임 전송.

2. **AWS 서버**:
   - 라즈베리파이로부터 프레임 수신.
   - 머신러닝 모델로 프레임 처리.
   - 처리된 프레임을 Flask를 통해 클라이언트로 스트리밍.

3. **클라이언트**:
   - AWS Flask 서버에 연결.
   - 처리된 비디오를 실시간으로 스트리밍 및 표시.

---
