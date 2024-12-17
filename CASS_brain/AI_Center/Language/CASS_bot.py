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


class CASS_BOT(nn.Module):
    def __init__(self):
        super(CASS_BOT, self).__init__()

        # daglo API 토큰
        self.HOST = 'https://apis.daglo.ai'
        self.API_TOKEN = 'm-Akio8rnh7zEfABZnCHACBO'

        self.recording = False
        self.audio_frames = []

        self.user_input = ""
        self.stt_order_list = []
        self.engine = False  # 시동 on, off 유무


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
                        result = result.replace('시동 거', '시동 꺼')
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

        #print("음성 인식 중...종료시 's' 키 다시 클릭")
        self.audio_frames = []

        while self.recording:
            data = stream.read(1024, exception_on_overflow=False)
            self.audio_frames.append(data)

        #print("음성 인식 완료")
        stream.stop_stream()
        stream.close()
        p.terminate()


    def handle_key(self, key):
        try:
            if key.char == 's': 
                if not self.recording:
                    self.recording = True
                    print("음성 인식 중...종료시 's' 키 클릭")
                    threading.Thread(target=self.record_audio).start()
                else:
                    self.recording = False
                    print("음성 인식 완료")

                    if self.audio_frames:
                        # Session Token 획득
                        res = self.get_session_token()
                        sid = res['sessionId']
                        s_token = res['sessionToken']

                        # 오디오 스트림 서버로 전송
                        self.send_audio_stream(b''.join(self.audio_frames), sid, s_token)

                        # STT 결과를 수신
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


    def first_result(self, input):
        chain = RemoteRunnable("https://notable-daily-sunbeam.ngrok-free.app/chat/")
        output = chain.invoke({"messages": [{"role": "user", "content": input}]})
        return output

    def check_result(self, output):
        self.flag = None  
        self.stt_order = None
        result = output
        # 시동 켜기
        if '시동' in result and ('걸' in result or '켜' in result):
            if ('off' in self.stt_order_list or self.stt_order_list == []):
                self.stt_order = 'on'
                self.engine = True
                if 'off' in self.stt_order_list:
                    off_idx = self.stt_order_list.index('off')
                    self.stt_order_list[off_idx] = 'on' 
                elif self.stt_order_list == []:
                    self.stt_order_list.append(self.stt_order)  
            elif 'on' in self.stt_order_list and self.engine == True:
                self.flag = 0
            else:
                pass
        # 시동 끄기
        elif '시동' in result and ('끄' in result or '꺼' in result):
            if 'on' in self.stt_order_list and not 'go' in self.stt_order_list:
                self.engine = False
                self.stt_order = 'off'
                on_idx = self.stt_order_list.index('on')
                self.stt_order_list[on_idx] = 'off'
                self.stt_order_list = [] 
            elif (self.stt_order_list == [] or 'off' in self.stt_order_list) and self.engine == False:
                self.flag = 1
            elif 'go' in self.stt_order_list:
                self.flag = 5
            else:
                pass
        # 주행 시작
        elif '출발' in result or ('주행' in result and '시작' in result):
            if 'on' in self.stt_order_list and self.engine == True: # 시동 켜져 있을 때
                if 'stop' in self.stt_order_list:
                    self.stt_order = 'go'
                    stop_idx = self.stt_order_list.index('stop')
                    self.stt_order_list[stop_idx] = 'go'
                elif not 'go' in self.stt_order_list:
                    self.stt_order = 'go'
                    self.stt_order_list.append(self.stt_order)
                else:
                    self.flag = 2
            else:
                self.flag = 4
        # 정차 하기
        elif '정차' in result or '정지' in result or '멈추' in result:
            if 'on' in self.stt_order_list:  # 시동 켜져 있을 때
                # self.stt_order = 'stop'
                if 'go' in self.stt_order_list:
                    self.stt_order = 'stop'
                    go_idx = self.stt_order_list.index('go')
                    self.stt_order_list[go_idx] = 'stop'
                elif self.stt_order_list == ['on'] or 'stop' in self.stt_order_list:
                    self.flag = 3
                else:
                    pass
            else:
                self.flag = 4
        else:
            pass

        if self.stt_order != None:
            print('order ---------------------> ', self.stt_order)
            return self.stt_order


    def cass_output(self, input):
        if '현재 시간' in input or '지금 시간' in input:
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
            output = f"현재 시간은 {ampm} {current_hour}시 {current_min}분 입니다."
        
        else:
            if self.flag != None:
                if self.flag == 0:
                    output = '이미 시동이 걸려있습니다.'
                elif self.flag == 1:
                    output = '이미 시동이 꺼져있습니다.'
                elif self.flag == 2:
                    output = '이미 주행 중 입니다.'
                elif self.flag == 3:
                    output = '이미 정차해 있는 상태입니다.'
                elif self.flag == 4:
                    output = '시동을 먼저 켜주세요.'
                else:
                    output = '주행 중 입니다. 차량을 정차시킨 후 시동을 꺼주세요.'
            else:
                output = input
        return output


    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='ko', slow=False)
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)  # TTS 결과 저장
            playsound(temp_file.name)  # 음성 재생
            self.user_input = ""


if __name__ == "__main__":
    # 모든 경고 무시
    # warnings.filterwarnings("ignore")   

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
                response = cass_bot.first_result(cass_bot.user_input)
                order = cass_bot.check_result(response)
                final_output = cass_bot.cass_output(response)
                print("CASS 응답:", final_output)
                print('=' * 50)
                cass_bot.text_to_speech(final_output)
            else:
                break