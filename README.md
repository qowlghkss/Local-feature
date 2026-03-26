# L04 Local Feature - Homework (컴퓨터비전)

> **과목**: 컴퓨터비전 | **교수**: 서정일 (동아대학교 컴퓨터AI공학부)  
> **주제**: Local Feature - SIFT 특징점 검출, 매칭, 호모그래피 정합

---

## 📁 레포지토리 구조

```
4week/
├── README.md                   ← 전체 레포지토리 설명 (현재 파일)
├── requirements.txt            ← 전체 공통 의존성 파일
│
├── base/                       ← 과제 제공 원본 이미지 (수정 불가)
│   ├── mot_color70.jpg
│   ├── mot_color83.jpg
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
│
├── Problem_1/                  ← 문제 1: SIFT 특징점 검출 및 시각화
│   ├── problem1.py             ← 코드 (모든 라인 한글 주석)
│   ├── requirements.txt        ← 문제별 의존성
│   ├── README.md               ← 문제별 상세 설명
│   └── output/                 ← 실행 결과 이미지 (자동 생성)
│       └── problem1_result.png
│
├── Problem_2/                  ← 문제 2: SIFT 두 영상 간 특징점 매칭
│   ├── problem2.py
│   ├── requirements.txt
│   ├── README.md
│   └── output/
│       └── problem2_result.png
│
└── Problem_3/                  ← 문제 3: 호모그래피 이미지 정합
    ├── problem3.py
    ├── requirements.txt
    ├── README.md
    └── output/
        ├── problem3_result.png
        ├── problem3_warped.png
        └── problem3_aligned.png
```

> 가상환경(`.venv/` 또는 conda env)은 `.gitignore`에 의해 GitHub에 업로드되지 않습니다.

---

## 🧪 문제 요약

| 문제 | 주제 | 입력 이미지 | 핵심 기술 |
|:----:|------|------------|---------|
| Problem 1 | SIFT 특징점 검출 및 시각화 | `mot_color70.jpg` | `SIFT_create`, `detectAndCompute`, `drawKeypoints` |
| Problem 2 | SIFT 두 영상 간 특징점 매칭 | `mot_color70.jpg` + `mot_color83.jpg` | `BFMatcher`, `knnMatch`, Lowe's Ratio Test |
| Problem 3 | 호모그래피 이미지 정합 | `img1.jpg` + `img2.jpg` | `findHomography`, `warpPerspective`, RANSAC |

---

## ⚙️ 빠른 시작 (Quick Start)

### 공통 가상환경 설정 (Python venv)

```bash
# 1. 최상위 디렉토리로 이동
cd /path/to/4week

# 2. 가상환경 생성
python3 -m venv .venv

# 3. 가상환경 활성화
source .venv/bin/activate   # Linux/macOS

# 4. 공통 패키지 설치
pip install -r requirements.txt
```

### 공통 가상환경 설정 (Conda)

```bash
# 1. conda 환경 생성 (Python 3.10)
conda create -n cv_hw python=3.10 -y
conda activate cv_hw

# 2. 패키지 설치
pip install -r requirements.txt
```

### 각 문제 실행

```bash
# Problem 1: SIFT 특징점 검출 및 시각화
cd Problem_1 && python problem1.py

# Problem 2: SIFT 두 영상 간 특징점 매칭
cd ../Problem_2 && python problem2.py

# Problem 3: 호모그래피 이미지 정합
cd ../Problem_3 && python problem3.py
```

---

## 📦 의존성 (Dependencies)

```
opencv-python==4.9.0.80
numpy==1.26.4
matplotlib==3.8.4
```

---

## 📌 참고사항

- 모든 Python 코드는 **모든 라인마다 한글 주석**이 포함되어 있습니다.
- 실행 결과 이미지는 각 폴더의 `output/` 디렉토리에 자동으로 저장됩니다.
- 과제에서 `mot_color80.jpg`를 명시하였으나, 제공된 파일 기준으로 **`mot_color83.jpg`** 를 사용합니다.
