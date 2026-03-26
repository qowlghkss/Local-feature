"""
문제 2: SIFT를 이용한 두 영상 간 특징점 매칭
과목: 컴퓨터비전 L04 Local Feature - Homework
교수: 서정일 (동아대학교 컴퓨터AI공학부)
"""

import cv2 as cv             # OpenCV 라이브러리 - SIFT 및 매칭 기능 제공
import matplotlib.pyplot as plt  # matplotlib - 매칭 결과 시각화 출력
import os                    # os - 파일 경로 처리

# ─────────────────────────────────────────────
# 1단계: 두 이미지 불러오기
# ─────────────────────────────────────────────
# 현재 스크립트 파일이 위치한 디렉토리의 절대 경로를 구함
script_dir = os.path.dirname(os.path.abspath(__file__))

# 과제에서 지정한 두 이미지 파일 경로 조합
# 이미지1: mot_color70.jpg (기준 이미지, Query Image)
# 이미지2: mot_color83.jpg (비교 대상 이미지, Train Image)
# ※ 과제에서 mot_color80.jpg라고 명시했으나 제공 파일은 mot_color83.jpg임
path1 = os.path.join(script_dir, '..', 'base', 'mot_color70.jpg')
path2 = os.path.join(script_dir, '..', 'base', 'mot_color83.jpg')

# cv.imread(): 이미지를 BGR 형식으로 읽어들임
img1_bgr = cv.imread(path1)   # 첫 번째 이미지 (mot_color70)
img2_bgr = cv.imread(path2)   # 두 번째 이미지 (mot_color83)

# 이미지를 정상적으로 불러왔는지 확인
if img1_bgr is None:
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path1}")
if img2_bgr is None:
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path2}")

# cv.cvtColor(): BGR → RGB 변환 (matplotlib 시각화용)
img1_rgb = cv.cvtColor(img1_bgr, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2_bgr, cv.COLOR_BGR2RGB)

# cv.cvtColor(): BGR → Grayscale 변환 (SIFT 특징점 검출용)
img1_gray = cv.cvtColor(img1_bgr, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2_bgr, cv.COLOR_BGR2GRAY)

print(f"[정보] 이미지1 로드 완료: {path1} | 크기: {img1_bgr.shape[:2]}")
print(f"[정보] 이미지2 로드 완료: {path2} | 크기: {img2_bgr.shape[:2]}")

# ─────────────────────────────────────────────
# 2단계: SIFT 특징점 검출 및 디스크립터 추출
# ─────────────────────────────────────────────
# cv.SIFT_create(): SIFT 검출기 초기화 (nfeatures=0 → 제한 없이 모든 특징점 검출)
sift = cv.SIFT_create(nfeatures=0)  # nfeatures=0이면 검출 개수 제한 없음

# detectAndCompute(): 두 이미지에 각각 특징점 검출 + 디스크립터 추출
# keypoints1, descriptors1: 이미지1의 특징점과 128차원 기술자(descriptor) 행렬
# keypoints2, descriptors2: 이미지2의 특징점과 128차원 기술자(descriptor) 행렬
keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

print(f"[결과] 이미지1 특징점 수: {len(keypoints1)}개")
print(f"[결과] 이미지2 특징점 수: {len(keypoints2)}개")

# ─────────────────────────────────────────────
# 3단계: BFMatcher + Lowe's Ratio Test로 특징점 매칭
# ─────────────────────────────────────────────
# cv.BFMatcher(): Brute-Force Matcher - 모든 특징점 쌍의 거리를 계산하여 가장 가까운 것을 매칭
# cv.NORM_L2: L2(유클리드) 거리를 사용 (SIFT 디스크립터는 실수형이므로 L2 사용)
# crossCheck=False: 양방향 확인 비활성화 → knnMatch()와 함께 사용하기 위해 False로 설정
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# knnMatch(): k-최근접 이웃 매칭 수행
# k=2 → 각 특징점에 대해 가장 가까운 2개의 이웃을 반환
# 반환값: DMatch 객체 쌍의 리스트 [[ best_match, second_best_match ], ...]
raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

print(f"[매칭] 초기 매칭 쌍 수: {len(raw_matches)}개")

# ─────────────────────────────────────────────
# 4단계: Lowe's Ratio Test로 좋은 매칭점만 선별
# ─────────────────────────────────────────────
# Lowe's Ratio Test: 최근접 이웃과 두 번째 이웃의 거리 비율을 비교
# 비율이 낮을수록 고유한 매칭 (다른 후보와 명확히 구분됨)
# 임계값 0.75: 비율이 0.75 미만인 경우에만 좋은 매칭으로 수락
RATIO_THRESHOLD = 0.75  # 비율 임계값 (논문에서 권장: 0.7~0.8)

good_matches = []  # 선별된 좋은 매칭점을 담을 리스트

for m, n in raw_matches:
    # m: 가장 가까운 이웃 (최근접 매칭)
    # n: 두 번째로 가까운 이웃
    # m.distance / n.distance 비율이 임계값보다 낮으면 신뢰할 수 있는 매칭으로 판단
    if m.distance < RATIO_THRESHOLD * n.distance:
        good_matches.append(m)  # 비율 조건을 통과한 매칭점 추가

print(f"[필터링] Ratio Test 후 좋은 매칭 수: {len(good_matches)}개 (임계값: {RATIO_THRESHOLD})")

# ─────────────────────────────────────────────
# 5단계: 매칭 결과 시각화
# ─────────────────────────────────────────────
# cv.drawMatches(): 두 이미지의 대응하는 특징점들 간에 선을 그어 매칭 시각화
# img1_rgb, keypoints1: 첫 번째 이미지와 그 특징점 리스트
# img2_rgb, keypoints2: 두 번째 이미지와 그 특징점 리스트
# good_matches: 시각화할 매칭 리스트
# None: 출력 이미지 (None → 자동 생성)
# flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS: 매칭되지 않은 단독 특징점은 표시 안 함
img_matches = cv.drawMatches(
    img1_rgb, keypoints1,   # 이미지1 + 특징점
    img2_rgb, keypoints2,   # 이미지2 + 특징점
    good_matches,           # 시각화할 매칭 리스트
    None,                   # 출력 이미지 (None → 새로 생성)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 매칭 안된 점 숨김
)

# matplotlib으로 매칭 결과 출력
plt.figure(figsize=(18, 6))  # 두 이미지를 나란히 보기 위해 가로로 넓게 설정
plt.suptitle(
    f'Problem 2: SIFT 특징점 매칭 결과\n'
    f'mot_color70.jpg ↔ mot_color83.jpg | '
    f'매칭 수: {len(good_matches)}개 (Ratio Test: {RATIO_THRESHOLD})',
    fontsize=13, fontweight='bold'
)
plt.imshow(img_matches)    # 매칭 선이 그려진 이미지 출력
plt.axis('off')            # 축(눈금) 제거
plt.tight_layout()         # 여백 자동 조정

# ─────────────────────────────────────────────
# 6단계: 결과 이미지 저장 및 화면 출력
# ─────────────────────────────────────────────
# 결과 이미지를 저장할 output 폴더 생성 (없으면 자동 생성)
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)  # exist_ok=True → 이미 있어도 오류 없음

# 매칭 결과 이미지 파일 경로
output_path = os.path.join(output_dir, 'problem2_result.png')

# plt.savefig(): 현재 figure를 파일로 저장 (dpi=150 → 해상도 설정)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[저장] 결과 이미지 저장 완료: {output_path}")

# plt.show(): 화면에도 팝업 창으로 표시 (GUI 환경에서 동작)
plt.show()

# ─────────────────────────────────────────────
# 7단계: 상위 10개 매칭 정보 출력 (분석용)
# ─────────────────────────────────────────────
# 매칭 결과를 거리 기준으로 오름차순 정렬 (거리가 작을수록 좋은 매칭)
good_matches_sorted = sorted(good_matches, key=lambda x: x.distance)

print("\n[매칭 상위 10개 정보 (거리 기준 오름차순)]")
print(f"{'순번':<5} {'이미지1 특징점 인덱스':>20} {'이미지2 특징점 인덱스':>20} {'매칭 거리':>12}")
print("-" * 60)
for i, match in enumerate(good_matches_sorted[:10]):  # 상위 10개 출력
    # match.queryIdx: 이미지1에서의 특징점 인덱스
    # match.trainIdx: 이미지2에서의 특징점 인덱스
    # match.distance: 두 특징점 디스크립터 간의 L2 거리 (낮을수록 더 유사)
    print(f"{i+1:<5} {match.queryIdx:>20} {match.trainIdx:>20} {match.distance:>12.2f}")
