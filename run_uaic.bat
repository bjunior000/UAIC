@echo off
chcp 65001 >nul
echo UAIC 프로젝트 실행 스크립트

REM 가상환경 활성화
call uaic_venv\Scripts\activate

echo.
echo 실행할 작업을 선택하세요:
echo 1. Bag-of-Words 모델 학습 (기존)
echo 2. Bag-of-Words 모델 학습 (CLIP 없이)
echo 3. 불확실성 계산 (모든 샘플)
echo 4. 불확실성 계산 (일부 샘플 - 빠른 테스트)
echo 5. 데이터 쌍 생성
echo 6. 모델 학습
echo 7. 모델 평가
echo 8. 종료
echo.

:MENU
set /p choice=번호를 입력하세요 (1-8): 

echo %choice%

if "%choice%"=="1" (
    echo Bag-of-Words 모델을 학습합니다...
    python bag_of_word.py --train
    goto MENU
)
if "%choice%"=="2" (
    echo CLIP 없이 Bag-of-Words 모델을 학습합니다...
    python bow_without_clip.py
    goto MENU
)
if "%choice%"=="3" (
    echo 불확실성을 계산합니다 - 모든 샘플...
    python bag_of_word.py --uncertainty
    goto MENU
)
if "%choice%"=="4" (
    echo 불확실성을 계산합니다 - 테스트용 - 100개 샘플
    python bag_of_word.py --uncertainty --max_samples 100 --batch_size 10
    goto MENU
)
if "%choice%"=="5" (
    echo 데이터 쌍을 생성합니다...
    python -c "from utilis import create_data_pair; create_data_pair('data')"
    goto MENU
)
if "%choice%"=="6" (
    echo 모델을 학습합니다...
    python train.py
    goto MENU
)
if "%choice%"=="7" (
    echo 모델을 평가합니다...
    python inference.py
    goto MENU
)
if "%choice%"=="8" (
    echo 프로그램을 종료합니다.
    goto END
)

echo 잘못된 선택입니다. 다시 시도하세요.
goto MENU

:END
deactivate
echo 가상환경이 비활성화되었습니다.
pause 