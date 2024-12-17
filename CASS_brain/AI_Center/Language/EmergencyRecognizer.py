import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pyaudio
import numpy as np

# 모델 정의 (간단한 CNN 모델)
class EmergencyRecognizer(nn.Module):
    def __init__(self, input_size=120, n_classes=2):
        super(EmergencyRecognizer, self).__init__()
        self.set_params(input_size=input_size, n_classes=n_classes)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 512, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1),
        )
        self.fc = nn.Linear(512, n_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 첫 번째 convolution 레이어
        x = torch.relu(self.conv2(x))  # 두 번째 convolution 레이어
        x = torch.relu(self.conv3(x))  # 두 번째 convolution 레이어
        x = x.mean(dim=2)  # Time axis 평균, 즉 각 채널의 평균값으로 차원 축소
        x = self.fc(x)  # Fully connected layer
        return x
    
    def set_params(self, input_size=120, n_classes=2, sample_rate=16000):
        # 파라미터 설정
        self.target_sample_rate = sample_rate  # 타겟 샘플 속도
        self.n_mfcc = input_size  # MFCC의 수
        self.n_classes = n_classes  # 구별해야 할 클래스 수 (사이렌 소리와 일반 소리)
        self.N2C = {0: 'Emergency', 1: 'Normal'}

    # 모델 훈련
    def train_model(self, filepaths, lr=0.001, epochs=15):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for filepath in filepaths:
                waveform = self.z_score_normalize(self.load_audio(filepath))
                mfcc_features = self.extract_features(waveform)
                # 모델 학습을 위한 데이터 형태로 변환 (배치 차원 추가)
                inputs = mfcc_features.cuda()  # (n_mfcc, time_steps) 형태로 변환
                label = 0 if "Siren" in filepath else 1  # 예시로 사이렌 파일명에 'sirens' 포함 여부로 라벨링
                label = torch.tensor([label], dtype=torch.long).cuda()  # 레이블을 배치 크기에 맞춰 변환
                optimizer.zero_grad()
                outputs = self.forward(inputs)  # (1, n_mfcc, time_steps)
                loss = criterion(outputs, label)  # 배치 크기 맞춰서 손실 계산
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(filepaths):.4f}")

    def z_score_normalize(self, waveform):
        # waveform의 평균과 표준편차 계산
        mean = waveform.mean()
        std = waveform.std()
        
        # Z-Score 정규화
        normalized_waveform = (waveform - mean) / std
        
        return normalized_waveform

    def load_audio(self, filepath, sample_rate=16000):
        waveform, sr = torchaudio.load(filepath)
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
        return waveform

    # MFCC 특징 추출
    def extract_features(self, waveform):
        mfcc_transform = transforms.MFCC(
            sample_rate=self.target_sample_rate,
            n_mfcc=self.n_mfcc,
            log_mels=True
        )
        mfcc = mfcc_transform(waveform)
        return mfcc

    def predict(self, path):
        waveform = self.load_audio(path)
        mfcc_features = self.extract_features(waveform[0])
        inputs = mfcc_features.unsqueeze(0)
        output = self.forward(inputs.cuda())
        result = self.N2C[output.argmax(-1).item()]
        return result
    
    def RTparameter_setting(self, channels=1, chunk=1024, buffer_size = 15):
        self.p = pyaudio.PyAudio()
        self.chunk = chunk

        # 마이크 입력을 받기 위한 설정
        format = pyaudio.paInt16  # 오디오 포맷
        input_device_index = None  # 기본 입력 장치 사용

        self.stream = self.p.open(format=format,
                        channels=channels,
                        rate=self.target_sample_rate,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=self.chunk)

        self.check = None
        self.frame_buffer = np.zeros((self.chunk*buffer_size))  # 1024 샘플을 버퍼에 저장 (모노 채널 가정)

    def RTsteaming(self, mode='qt'):
        if mode=='qt':
            # 오디오 청크 읽기
            audio_data = self.stream.read(self.chunk)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # 버퍼에 추가된 오디오 데이터
            self.frame_buffer = np.roll(self.frame_buffer, -self.chunk)  # 이전 데이터는 뒤로 밀고
            self.frame_buffer[-self.chunk:] = audio_np  # 현재 데이터를 버퍼의 끝에 추가
            # 오디오 데이터가 1초 이상 축적되면 예측
            result = None
            if np.count_nonzero(self.frame_buffer) > 0:  # 버퍼가 비어있지 않으면
                waveform = self.z_score_normalize(torch.from_numpy(self.frame_buffer).float())
                mfcc_features = self.extract_features(waveform)
                inputs = mfcc_features.unsqueeze(0)
                output = self.forward(inputs.cuda())
                result = self.N2C[output.argmax(-1).item()]       
            return result
        else:     
            try:
                while True:
                    # 오디오 청크 읽기
                    audio_data = self.stream.read(self.chunk)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)

                    # 버퍼에 추가된 오디오 데이터
                    self.frame_buffer = np.roll(self.frame_buffer, -self.chunk)  # 이전 데이터는 뒤로 밀고
                    self.frame_buffer[-self.chunk:] = audio_np  # 현재 데이터를 버퍼의 끝에 추가
                    # 오디오 데이터가 1초 이상 축적되면 예측
                    if np.count_nonzero(self.frame_buffer) > 0:  # 버퍼가 비어있지 않으면
                        waveform = self.z_score_normalize(torch.from_numpy(self.frame_buffer).float())
                        mfcc_features = self.extract_features(waveform)
                        inputs = mfcc_features.unsqueeze(0)
                        output = self.forward(inputs.cuda())
                        result = self.N2C[output.argmax(-1).item()]

                        # 결과 출력 (변경이 있을 때만)
                        if self.check != result:
                            print(f"Detected: {result}")
                            self.check = result
                
            except KeyboardInterrupt:
                print("실시간 소리 판별 테스트 종료.")

            finally:
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
    

