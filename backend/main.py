import io
import uuid
import wave
import os
import subprocess
import numpy as np
from typing import Union
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper 모델 설정
model_size = "medium"
device = "cuda"
compute_type = "float16"
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# FFmpeg 경로 (절대 경로로 명시)
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    input_path = f"temp_input_{file_id}.wav"
    audio_path = f"temp_processed_{file_id}.wav"

    try:
        print(f"[DEBUG] 파일 저장 중: {input_path}")
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        if not os.path.exists(input_path):
            print("[ERROR] 업로드된 파일 저장 실패")
            return {"error": "파일 저장 실패"}

        print(f"[DEBUG] FFmpeg 변환 시작: {input_path} -> {audio_path}")
        result = subprocess.run([
            ffmpeg_path, "-y", "-i", input_path,
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            audio_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print("[FFmpeg 오류 발생]")
            print(result.stderr)
            raise RuntimeError("FFmpeg 변환 실패")

        print("[DEBUG] Whisper 모델 실행")
        segments, _ = model.transcribe(audio_path)
        text = " ".join(segment.text for segment in segments)

        print(f"[DEBUG] 인식 결과: {text}")
        return {"transcription": text}

    except Exception as e:
        print(f"[ERROR] 오디오 처리 중 오류 발생: {str(e)}")
        return {"error": str(e)}, 500

    finally:
        for path in [input_path, audio_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    print(f"[WARN] 파일 삭제 실패: {path}")

@app.websocket("/ws/transcribe/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 클라이언트 연결됨")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"[DEBUG] 오디오 데이터 수신: {len(data)} 바이트")

            try:
                temp_file = f"temp_ws_{uuid.uuid4().hex}.raw"
                wav_file = temp_file.replace(".raw", ".wav")

                with open(temp_file, "wb") as f:
                    f.write(data)

                if len(data) % 2 != 0:
                    data += b'\0'

                audio_data = np.frombuffer(data, dtype=np.int16)

                with wave.open(wav_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data.tobytes())

                if os.path.getsize(wav_file) < 1000:
                    print("오디오 파일이 너무 작습니다.")
                    continue

                segments, _ = model.transcribe(wav_file)
                text = " ".join(segment.text for segment in segments)
                print(f"[DEBUG] 실시간 인식 결과: {text}")

                if text.strip():
                    await websocket.send_text(text)

            except Exception as process_error:
                print(f"[ERROR] WebSocket 오디오 처리 오류: {process_error}")

            finally:
                for path in [temp_file, wav_file]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            print(f"[WARN] WebSocket 파일 삭제 실패: {path}")

    except WebSocketDisconnect:
        print("❌ 클라이언트 연결 종료")
    except Exception as e:
        print(f"[ERROR] WebSocket 예외: {e}")
