"""
문제 3: 호모그래피를 이용한 이미지 정합 (Image Alignment)
과목: 컴퓨터비전 L04 Local Feature - Homework
교수: 서정일 (동아대학교 컴퓨터AI공학부)
"""

import cv2 as cv              # OpenCV - SIFT, 호모그래피, 원근 변환 기능 제공
import numpy as np            # NumPy - 행렬 연산 및 이미지 배열 처리
import matplotlib.pyplot as plt  # matplotlib - 최종 결과 시각화
import os                     # os - 파일 경로 처리

# ─────────────────────────────────────────────
# 1단계: 두 이미지 불러오기 (img1.jpg, img2.jpg)
# ─────────────────────────────────────────────
# 현재 스크립트 파일이 위치한 디렉토리의 절대 경로를 구함
script_dir = os.path.dirname(os.path.abspath(__file__))

# 과제에서 지정한 두 이미지 경로 조합 (img1.jpg와 img2.jpg 선택)
path1 = os.path.join(script_dir, '..', 'base', 'img1.jpg')  # 변환 기준 이미지
path2 = os.path.join(script_dir, '..', 'base', 'img2.jpg')  # 변환 대상 이미지

# cv.imread(): 이미지를 BGR 형식으로 읽어들임
img1_bgr = cv.imread(path1)   # 기준 이미지 (Destination) - 정합의 목표 좌표계
img2_bgr = cv.imread(path2)   # 변환 이미지 (Source) - 이 이미지를 img1에 맞게 변환

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

# 각 이미지의 너비(w)와 높이(h) 추출 (파노라마 출력 크기 계산에 사용)
h1, w1 = img1_bgr.shape[:2]   # 이미지1의 (높이, 너비)
h2, w2 = img2_bgr.shape[:2]   # 이미지2의 (높이, 너비)

print(f"[정보] 이미지1 로드: {path1} | 크기: {h1}x{w1}")
print(f"[정보] 이미지2 로드: {path2} | 크기: {h2}x{w2}")

# ─────────────────────────────────────────────
# 2단계: SIFT 특징점 검출 및 디스크립터 추출
# ─────────────────────────────────────────────
# cv.SIFT_create(): SIFT 검출기 초기화
# nfeatures=0 → 검출 개수 제한 없음 (호모그래피 정확도를 위해 최대한 많은 특징점 필요)
sift = cv.SIFT_create(nfeatures=0)

# detectAndCompute(): 두 이미지에서 각각 특징점과 디스크립터 동시 추출
keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

print(f"[SIFT] 이미지1 특징점 수: {len(keypoints1)}개")
print(f"[SIFT] 이미지2 특징점 수: {len(keypoints2)}개")

# ─────────────────────────────────────────────
# 3단계: BFMatcher + knnMatch + Lowe's Ratio Test로 매칭점 선별
# ─────────────────────────────────────────────
# cv.BFMatcher(): Brute-Force Matcher 초기화
# cv.NORM_L2: SIFT는 실수형 디스크립터이므로 L2(유클리드) 거리 사용
# crossCheck=False: knnMatch()와 함께 사용이기 때문에 False 설정
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# knnMatch(): 각 특징점에 대해 k=2개의 최근접 이웃을 찾음
# 두 개를 찾는 이유: Lowe's Ratio Test를 위해 1등과 2등을 비교하기 위함
raw_matches = bf.knnMatch(descriptors2, descriptors1, k=2)
# ※ 주의: descriptors2를 query, descriptors1을 train으로 설정
#   → 이미지2의 각 특징점에서 이미지1의 대응점을 찾는 방향으로 매칭

# Lowe's Ratio Test: 비율 임계값(0.7)보다 작은 경우만 선별 (신뢰도 높은 매칭)
RATIO_THRESHOLD = 0.7   # 논문에서 권장하는 표준 임계값 (0.7)
good_matches = []       # 선별된 좋은 매칭을 담을 리스트

for m, n in raw_matches:
    # m: 이미지1에서의 가장 가까운 매칭
    # n: 이미지1에서의 두 번째로 가까운 매칭
    # 비율이 낮을수록 m은 n보다 훨씬 가까운 고유한 매칭으로 신뢰할 수 있음
    if m.distance < RATIO_THRESHOLD * n.distance:
        good_matches.append(m)  # 좋은 매칭 리스트에 추가

print(f"[매칭] Ratio Test 통과 매칭 수: {len(good_matches)}개 (임계값: {RATIO_THRESHOLD})")

# 호모그래피 계산을 위해 최소 4개의 대응점이 필요
if len(good_matches) < 4:
    raise ValueError(f"좋은 매칭 수({len(good_matches)})가 너무 적습니다. 임계값을 조정하세요.")

# ─────────────────────────────────────────────
# 4단계: 호모그래피 행렬 계산 (RANSAC 이상점 제거 포함)
# ─────────────────────────────────────────────
# 이미지2(src)와 이미지1(dst)의 매칭 특징점 좌표를 NumPy 배열로 추출
# keypoints2[m.queryIdx].pt: 이미지2에서의 특징점 좌표 (x, y)
# keypoints1[m.trainIdx].pt: 이미지1에서의 대응 특징점 좌표 (x, y)
# np.float32: 부동소수점 형식으로 변환 (호모그래피 계산에 필요)
src_pts = np.float32([keypoints2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# reshape(-1, 1, 2): cv.findHomography()의 입력 형식에 맞춰 (N, 1, 2) 형태로 변환

# cv.findHomography(): 두 이미지 간의 호모그래피 행렬(3x3) 계산
# src_pts → 이미지2의 특징점 좌표 (변환 출발점)
# dst_pts → 이미지1의 대응 특징점 좌표 (변환 도착점)
# cv.RANSAC: Random Sample Consensus 알고리즘 사용 → 이상점(Outlier) 강건하게 제거
# 5.0: RANSAC 허용 오차 픽셀 (이 픽셀 내에 있으면 inlier로 판단)
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
# H: 3x3 호모그래피 행렬 (이미지2 → 이미지1 좌표계 변환에 사용)
# mask: 각 매칭점이 inlier(1)인지 outlier(0)인지 나타내는 마스크 배열

# inlier(이상점이 아닌 정상 매칭점) 수 계산
inliers = int(mask.sum())
print(f"[호모그래피] 계산 완료 | 총 매칭: {len(good_matches)}개 | Inlier: {inliers}개")
print(f"[호모그래피 행렬 H]:\n{H}")

# ─────────────────────────────────────────────
# 5단계: 호모그래피 적용 - 이미지2를 이미지1에 정합
# ─────────────────────────────────────────────
# 파노라마(정합) 결과 이미지의 출력 크기를 두 이미지를 합친 크기로 설정
# 너비: w1 + w2 (두 이미지를 가로로 합친 너비)
# 높이: max(h1, h2) (두 이미지 중 더 큰 높이 사용)
out_w = w1 + w2    # 출력 이미지 너비
out_h = max(h1, h2)  # 출력 이미지 높이

# cv.warpPerspective(): 호모그래피 행렬 H를 이용해 이미지2를 원근 변환
# img2_bgr: 변환할 이미지
# H: 변환에 사용할 호모그래피 행렬
# (out_w, out_h): 출력 이미지의 크기 (너비, 높이)
warped_img2 = cv.warpPerspective(img2_bgr, H, (out_w, out_h))
# warped_img2: 이미지1의 좌표계(시점)로 변환된 이미지2

# 변환된 이미지(warped_img2) 위에 이미지1을 덮어쓰기 (정합 결과 합성)
# warped_img2의 왼쪽 상단에 img1_bgr을 복사하여 합치기
result_bgr = warped_img2.copy()      # 변환된 이미지를 기초 캔버스로 복사
result_bgr[0:h1, 0:w1] = img1_bgr   # 이미지1 영역에 원본 이미지1을 그대로 붙여넣기

# BGR → RGB 변환 (matplotlib 시각화용)
warped_rgb = cv.cvtColor(warped_img2, cv.COLOR_BGR2RGB)  # 변환된 이미지2 (RGB)
result_rgb = cv.cvtColor(result_bgr, cv.COLOR_BGR2RGB)   # 최종 정합 결과 (RGB)

print("[정합] 이미지 변환 및 합성 완료")

# ─────────────────────────────────────────────
# 6단계: 특징점 매칭 결과 시각화 (Inlier만 표시)
# ─────────────────────────────────────────────
# Inlier 매칭만 선별하여 더 깔끔한 시각화
# mask.ravel(): 2D 마스크를 1D로 펼침 → 각 매칭이 inlier(1)인지 확인
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask.ravel()[i] == 1]

# cv.drawMatches(): 두 이미지와 매칭 선을 함께 그려 시각화
# ※ 매칭 방향: img2 → img1 순서로 그리기 (queryIdx=img2, trainIdx=img1)
img_matches_vis = cv.drawMatches(
    img2_rgb, keypoints2,    # Source 이미지 (img2) + 특징점
    img1_rgb, keypoints1,    # Destination 이미지 (img1) + 특징점
    inlier_matches[:50],     # 상위 50개 inlier 매칭만 표시 (가독성)
    None,                    # 출력 이미지 (None → 자동 생성)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 단독 특징점 숨김
)

# ─────────────────────────────────────────────
# 7단계: matplotlib으로 전체 결과 시각화 (2행 2열 레이아웃)
# ─────────────────────────────────────────────
# 2행 2열 서브플롯 구성으로 4가지 이미지를 한 번에 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Problem 3: 호모그래피를 이용한 이미지 정합 (Image Alignment)',
             fontsize=15, fontweight='bold')

# 서브플롯 (0,0): 원본 이미지1 (기준 이미지)
axes[0, 0].imshow(img1_rgb)                        # 이미지1 표시
axes[0, 0].set_title('이미지1 (img1.jpg)\n기준 이미지', fontsize=11)  # 제목
axes[0, 0].axis('off')                             # 축 제거

# 서브플롯 (0,1): 원본 이미지2 (변환 대상 이미지)
axes[0, 1].imshow(img2_rgb)                        # 이미지2 표시
axes[0, 1].set_title('이미지2 (img2.jpg)\n변환 대상 이미지', fontsize=11)
axes[0, 1].axis('off')

# 서브플롯 (1,0): 특징점 매칭 결과 (Inlier)
axes[1, 0].imshow(img_matches_vis)                 # 매칭 결과 시각화
axes[1, 0].set_title(
    f'특징점 매칭 결과\n(Inlier: {inliers}개, 상위 50개 표시)', fontsize=11
)
axes[1, 0].axis('off')

# 서브플롯 (1,1): 최종 정합 결과 (이미지2를 이미지1에 정렬)
axes[1, 1].imshow(result_rgb)                      # 정합 결과 이미지 표시
axes[1, 1].set_title('최종 정합 결과\n(Warped Image + img1)', fontsize=11)
axes[1, 1].axis('off')

plt.tight_layout()  # 서브플롯 간 여백 자동 조정

# ─────────────────────────────────────────────
# 8단계: 결과 저장 및 화면 출력
# ─────────────────────────────────────────────
# 결과 이미지를 저장할 output 폴더 생성 (없으면 자동 생성)
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)  # exist_ok=True → 이미 있어도 에러 없음

# 결합된 시각화 결과를 파일로 저장
result_fig_path = os.path.join(output_dir, 'problem3_result.png')
plt.savefig(result_fig_path, dpi=150, bbox_inches='tight')
print(f"[저장] 전체 결과 이미지 저장 완료: {result_fig_path}")

# 변환(Warped)된 이미지만 별도로 저장
warped_path = os.path.join(output_dir, 'problem3_warped.png')
cv.imwrite(warped_path, warped_img2)   # BGR 형식으로 저장 (OpenCV의 기본 형식)
print(f"[저장] Warped 이미지 저장 완료: {warped_path}")

# 정합 결과 이미지만 별도로 저장
aligned_path = os.path.join(output_dir, 'problem3_aligned.png')
cv.imwrite(aligned_path, result_bgr)   # BGR 형식으로 저장
print(f"[저장] 정합 결과 이미지 저장 완료: {aligned_path}")

# plt.show(): 화면에 팝업 창으로 표시 (GUI 환경에서 동작)
plt.show()

# ─────────────────────────────────────────────
# 9단계: 호모그래피 행렬 상세 분석 출력
# ─────────────────────────────────────────────
print("\n[호모그래피 행렬 상세 분석]")
print("H (3x3 변환 행렬):")
for row in H:
    # H는 3x3 행렬로, 이미지2의 픽셀 좌표를 이미지1의 좌표계로 변환하는 데 사용됨
    # H[0,0], H[0,1]: 회전 및 스케일 요소
    # H[0,2], H[1,2]: 평행 이동(translation) 요소
    # H[2,0], H[2,1]: 원근(perspective) 변환 요소
    # H[2,2]: 동차 좌표계의 정규화 요소 (보통 1에 가까운 값)
    print(f"  {row[0]:>12.6f}  {row[1]:>12.6f}  {row[2]:>12.6f}")
print(f"\n[요약] 전체 매칭: {len(good_matches)}개 → RANSAC Inlier: {inliers}개 "
      f"({100*inliers/len(good_matches):.1f}% 정확도)")
