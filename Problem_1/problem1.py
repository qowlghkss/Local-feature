"""
문제 1: SIFT를 이용한 특징점 검출 및 시각화

"""

import cv2 as cv          # OpenCV 라이브러리 - 컴퓨터비전 핵심 기능 제공
import matplotlib.pyplot as plt  # matplotlib - 이미지 시각화 출력을 위한 라이브러리
import os                 # os - 파일 경로 처리를 위한 표준 라이브러리

# ─────────────────────────────────────────────
# 1단계: 이미지 불러오기
# ─────────────────────────────────────────────
# 현재 스크립트 파일이 위치한 디렉토리의 절대 경로를 구함
script_dir = os.path.dirname(os.path.abspath(__file__))

# 과제에서 지정한 이미지 파일 경로를 조합
# → 상위 폴더(4week) 내의 base/ 폴더에 있는 mot_color70.jpg 사용
image_path = os.path.join(script_dir, '..', 'base', 'mot_color70.jpg')

# cv.imread(): 이미지를 BGR 형식으로 읽어들임 (OpenCV의 기본 채널 순서)
img_bgr = cv.imread(image_path)

# 이미지를 제대로 불러왔는지 확인 (None이면 파일 없음 또는 경로 오류)
if img_bgr is None:
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

# cv.cvtColor(): BGR → RGB 변환 (matplotlib은 RGB 형식으로 출력)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# cv.cvtColor(): BGR → Grayscale 변환 (SIFT는 그레이스케일 이미지에서 동작)
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

print(f"[정보] 이미지 로드 완료: {image_path}")
print(f"[정보] 이미지 크기 (H x W): {img_bgr.shape[:2]}")

# ─────────────────────────────────────────────
# 2단계: SIFT 객체 생성 및 특징점 검출
# ─────────────────────────────────────────────
# cv.SIFT_create(): SIFT 특징점 검출기를 초기화
# nfeatures=500 → 검출할 최대 특징점 수를 500개로 제한 (너무 많은 특징점 방지)
# nOctaveLayers=3 → 각 옥타브(스케일 단계)에서 사용할 레이어 수
# contrastThreshold=0.04 → 대비(contrast)가 낮은 약한 특징점을 제거하는 임계값
# edgeThreshold=10 → 엣지 응답이 강한 불안정한 특징점을 제거하는 임계값
# sigma=1.6 → 첫 번째 옥타브의 가우시안 블러 표준편차
sift = cv.SIFT_create(
    nfeatures=500,         # 최대 특징점 수 제한
    nOctaveLayers=3,       # 옥타브당 레이어 수
    contrastThreshold=0.04,# 대비 임계값 (낮을수록 더 많은 특징점 검출)
    edgeThreshold=10,      # 엣지 임계값
    sigma=1.6              # 가우시안 블러 시그마 값
)

# detectAndCompute(): 특징점 검출(detect)과 디스크립터 추출(compute) 동시 수행
# 반환값:
#   keypoints → 검출된 특징점들의 리스트 (위치, 크기, 방향 등 포함)
#   descriptors → 각 특징점의 128차원 특징 벡터 행렬 (매칭에 사용)
# None → 마스크 없이 전체 이미지에서 특징점 검출
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# 검출된 특징점 수 출력
print(f"[결과] 검출된 특징점 수: {len(keypoints)}개")

# ─────────────────────────────────────────────
# 3단계: 특징점 시각화 (풍부한 키포인트 표시)
# ─────────────────────────────────────────────
# cv.drawKeypoints(): 원본 이미지에 특징점을 그려넣는 함수
# img_rgb → 특징점을 그릴 원본 이미지 (RGB)
# keypoints → 시각화할 특징점들의 리스트
# None → 출력 이미지 (None이면 새 이미지를 생성)
# flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS:
#   → 특징점의 위치뿐만 아니라 크기(원의 반지름)와 방향(선의 각도)도 함께 시각화
img_keypoints = cv.drawKeypoints(
    img_rgb,          # 입력 이미지 (RGB 형식)
    keypoints,        # 시각화할 특징점 리스트
    None,             # 출력 이미지 (None → 새로 생성)
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 크기 및 방향 포함하여 그리기
)

# ─────────────────────────────────────────────
# 4단계: matplotlib으로 원본 이미지와 특징점 이미지를 나란히 출력
# ─────────────────────────────────────────────
# plt.figure(): 새로운 figure(그래프 창)를 생성, figsize로 전체 크기(인치) 지정
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 전체 제목 설정 (fontsize: 글자 크기, fontweight: 굵기)
fig.suptitle('Problem 1: SIFT 특징점 검출 및 시각화', fontsize=16, fontweight='bold')

# 첫 번째 서브플롯: 원본 이미지 출력
axes[0].imshow(img_rgb)                          # RGB 원본 이미지 표시
axes[0].set_title('원본 이미지\n(Original Image)', fontsize=13)  # 서브플롯 제목
axes[0].axis('off')                              # 축(눈금) 제거

# 두 번째 서브플롯: 특징점이 그려진 이미지 출력
axes[1].imshow(img_keypoints)                    # 특징점 시각화 이미지 표시
axes[1].set_title(
    f'SIFT 특징점 시각화\n(검출된 특징점 수: {len(keypoints)}개)', fontsize=13
)                                                # 특징점 수도 제목에 포함
axes[1].axis('off')                              # 축(눈금) 제거

# 서브플롯 간 여백 자동 조정 (겹치지 않도록)
plt.tight_layout()

# ─────────────────────────────────────────────
# 5단계: 결과 이미지 저장 및 화면 출력
# ─────────────────────────────────────────────
# 결과 저장 디렉토리 생성 (없으면 자동 생성)
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)  # exist_ok=True → 이미 존재해도 에러 없음

# 결과 이미지 파일 경로 지정
output_path = os.path.join(output_dir, 'problem1_result.png')

# plt.savefig(): 현재 figure를 파일로 저장
# dpi=150 → 해상도 설정 (dots per inch, 높을수록 선명)
# bbox_inches='tight' → 여백을 자동으로 조정하여 잘림 방지
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[저장] 결과 이미지가 저장되었습니다: {output_path}")

# plt.show(): 저장된 이미지를 화면에도 출력 (GUI 환경에서 팝업 창으로 표시)
plt.show()

# ─────────────────────────────────────────────
# 6단계: 상위 10개 특징점 정보 출력 (디버깅/분석 용도)
# ─────────────────────────────────────────────
print("\n[특징점 상위 10개 정보]")
print(f"{'순번':<5} {'X좌표':>8} {'Y좌표':>8} {'크기':>8} {'방향(°)':>10} {'응답값':>12}")
print("-" * 55)
for i, kp in enumerate(keypoints[:10]):  # 상위 10개만 출력
    # kp.pt: 특징점의 (x, y) 좌표 (소수점 포함)
    # kp.size: 특징점 이웃 영역의 반지름 (클수록 더 큰 스케일에서 감지됨)
    # kp.angle: 특징점의 주 방향 (0~360도, -1이면 계산 안 됨)
    # kp.response: 특징점의 강도(품질) 값 (클수록 더 좋은 특징점)
    print(f"{i+1:<5} {kp.pt[0]:>8.1f} {kp.pt[1]:>8.1f} {kp.size:>8.1f} {kp.angle:>10.1f} {kp.response:>12.4f}")
