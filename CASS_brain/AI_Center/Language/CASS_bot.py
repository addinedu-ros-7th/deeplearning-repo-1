# ollama 모델 및 langserve 관련 모듈
# from langchain_ollama import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# ---------------------------------------------------------------------------------
# remote 로 사용시 해당 모듈만 import 해도 됨.
from langserve import RemoteRunnable
import torch.nn as nn

# tts 를 위한 모듈
from gtts import gTTS
from playsound import playsound
import tempfile  # 임시 파일 생성

# stt 를 위한 모듈
import base64
import json
import pyaudio
import requests
from requests_sse import EventSource, InvalidStatusCodeError, InvalidContentTypeError
from pynput import keyboard
import threading
import re

from datetime import datetime
import warnings

# 모든 경고 무시
warnings.filterwarnings("ignore")  

class CASS_BOT(nn.Module):
    def __init__(self):
        super(CASS_BOT, self).__init__()

        # daglo API 토큰
        self.HOST = 'https://apis.daglo.ai'
        self.API_TOKEN = 'm-Akio8rnh7zEfABZnCHACBO'

        self.recording = False
        self.audio_frames = []

        self.user_input = ""


    def get_session_token(self):
        headers = {'Authorization': f'Bearer {self.API_TOKEN}'}
        return requests.get(self.HOST + "/stt/v1/stream/sessionTokens", headers=headers).json()


    def send_audio_stream(self, audio_stream: bytes, sid: str, s_token: str):
        headers = {'Authorization': f'Bearer {s_token}', 'Content-Type': 'application/json'}
        audio_str = base64.b64encode(audio_stream).decode('utf-8')
        data = {'sessionId': sid, 'channel': 0, 'file': audio_str}

        # 서버로 오디오 전송 후 응답을 확인
        try:
            res = requests.post(self.HOST + "/stt/v1/stream/recognize", headers=headers, json=data)
            if res.status_code == 200:
                print("Audio sent successfully")
            else:
                print(f"Failed to send audio. Status code: {res.status_code}")
        except Exception as e:
            print(f"Error sending audio: {e}")

        requests.post(self.HOST + "/stt/v1/stream/recognize", headers=headers, json=data)


    def get_audio_stream(self, sid: str, s_token: str):
        headers = {'Authorization': f'Bearer {s_token}'}
        with EventSource(self.HOST + f"/stt/v1/stream/sse/{sid}", timeout=30, headers=headers) as event_source:
            try:
                for event in event_source:
                    data = json.loads(event.data)
                    result = data.get('sttResult', {}).get('transcript', '')
                    if result:
                        result = re.sub(r'카스|갔을|갔어|가스', 'CASS', result)
                        print('=' * 50)
                        print(f"STT Result: {result}")
                        return result
            except InvalidStatusCodeError:
                pass
            except InvalidContentTypeError:
                pass
            except requests.RequestException:
                pass

    def record_audio(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)

        print("Recording started. Press 's' to stop.")
        self.audio_frames = []

        while self.recording:
            data = stream.read(1024, exception_on_overflow=False)
            self.audio_frames.append(data)

        print("Recording stopped.")
        stream.stop_stream()
        stream.close()
        p.terminate()


    def handle_key(self, key):
        try:
            if key.char == 's':  # Start/Stop recording
                if not self.recording:
                    self.recording = True
                    print("Starting recording...")
                    threading.Thread(target=self.record_audio).start()
                else:
                    self.recording = False
                    print("Stopping recording...")

                    if self.audio_frames:
                        # 2. Session Token을 획득
                        res = self.get_session_token()
                        sid = res['sessionId']
                        s_token = res['sessionToken']

                        # 3. 오디오 스트림을 서버로 전송
                        self.send_audio_stream(b''.join(self.audio_frames), sid, s_token)

                        # 4. 동시에 실시간 STT 결과를 수신
                        stt_result = self.get_audio_stream(sid, s_token)

                        if stt_result:
                            self.user_input = stt_result

                        # 전송 후 초기화
                        self.audio_frames = []
                        return False
            elif key.char == 'q':  # Quit
                print("Exiting program...")
                return False
        except AttributeError:
            pass


    def cass_result(self, input):
        chain = RemoteRunnable("https://notable-daily-sunbeam.ngrok-free.app/chat/")
        cass_output = chain.invoke({"messages": [{"role": "user", "content": input}]})
        return cass_output

    def check_result(self, output):
        self.stt_order = None
        result = output
        # 시동 켜기
        if '시동' in result and ('걸' in result or '켜' in result):
            self.stt_order = 'on'
        # 시동 끄기
        elif '시동' in result and ('끄' in result or '꺼' in result):
            self.stt_order = 'off'
        # 주행 시작
        elif '출발' in result or ('주행' in result and '시작' in result):
            self.stt_order = 'go'
        # 정차 하기
        elif '정차' in result or '정지' in result or '멈추' in result:
            self.stt_order = 'stop'
        else:
            pass
        return self.stt_order


    def calc_time(self):
        now = datetime.now()
        ampm = None
        current_hour = now.hour
        if current_hour >= 12:  # 오후
            if current_hour == 12:
                current_hour = current_hour
            else:
                current_hour = current_hour - 12
            ampm = '오후'
        else:  # 오전
            current_hour = current_hour
            ampm = '오전'
        current_min = now.minute
        current_time = f"현재 시간은 {ampm} {current_hour}시 {current_min}분 입니다."
        return current_time


    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='ko', slow=False)
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)  # TTS 결과 저장
            playsound(temp_file.name)  # 음성 재생
            self.user_input = ""


if __name__ == "__main__": 
    print("'s' 키를 눌러 음성 명령, 다시 's' 키를 눌러 명령 종료")
    print("시스템 종료는 'q' 키를 누르세요.")
    print("=" * 50)

    cass_bot = CASS_BOT()

    while True:
        with keyboard.Listener(on_press=cass_bot.handle_key) as listener:
            listener.join()
            if cass_bot.user_input:
                print('CASS_bot activated!')
                print('user_input : ', cass_bot.user_input)
                response = cass_bot.cass_result(cass_bot.user_input)
                
                order = cass_bot.check_result(response)
                if order != None:
                    print('order ------------------->', order)

                if '현재 시간은' in response:
                    response = cass_bot.calc_time()

                # 모델 응답 출력
                print("CASS 응답:", response)
                print('=' * 50)

                cass_bot.text_to_speech(response)
            
            else:
                break
