@echo off
chcp 65001 >nul
echo UAIC 프로젝트를 위한 가상환경 설정을 시작합니다...

REM 필요한 디렉토리 생성
if not exist "model" mkdir model
if not exist "data" mkdir data
if not exist "data\annotations" mkdir data\annotations

REM 가상환경 생성
python -m venv uaic_venv
call uaic_venv\Scripts\activate

REM 필요한 패키지 설치
echo 필요한 패키지를 설치합니다...
pip install --upgrade pip
pip install torch torchvision transformers nltk h5py numpy tqdm matplotlib pillow scipy

echo CLIP 패키지를 직접 GitHub에서 설치합니다...
pip install git+https://github.com/openai/CLIP.git

echo 가상환경 설정이 완료되었습니다.
echo.
echo 실행 방법:
echo 1. 가상환경 활성화: call uaic_venv\Scripts\activate
echo 2. BOW 모델 학습: python bag_of_word.py
echo.
echo 주의: coco_detections.hdf5 파일과 annotations를 data 폴더에 다운로드하세요.
echo - https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view (annotations.zip)
echo - https://drive.google.com/open?id=1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx (coco_detections.hdf5)
echo.
pause 