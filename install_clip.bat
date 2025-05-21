@echo off
echo CLIP 패키지를 Windows 환경에서 설치합니다...
call uaic_venv\Scripts\activate

REM 설치 명령어 실행
pip install git+https://github.com/openai/CLIP.git

echo CLIP 설치가 완료되었습니다.
pause 