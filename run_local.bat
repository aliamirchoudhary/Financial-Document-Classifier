@echo off
REM Local dev runner for the Pipeline-2 classifier service (uses the vert venv).
REM Usage:  run_local.bat [port]
setlocal

set VENV=D:\Project\Vertekx\2\vert\Scripts\python.exe
set PORT=%1
if "%PORT%"=="" set PORT=8000

REM Point the service at the checkpoint + corpus (edit if paths change)
set MODEL_DIR=D:\Project\Vertekx\pipeline 2
set CORPUS_JSONL=D:\Project\Vertekx\pipeline 2\data\s50r\colab.jsonl

cd /d "%~dp0"
"%VENV%" -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%