import os
import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pyaudio
import numpy as np
import sounddevice as sd
import warnings

# 모든 경고 무시
warnings.filterwarnings("ignore")

from scipy.io.wavfile import write
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 모델 정의 (간단한 CNN 모델)
class VoiceRecognizer(nn.Module):
    def __init__(self, input_size, n_classes=6):
        super(VoiceRecognizer, self).__init__()
        self.set_params(input_size=input_size, n_classes=n_classes, sample_rate=16000)

        self.voice_encoder1 = nn.Sequential(
            nn.Conv1d(input_size, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.voice_encoder2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        self.dense_encode = nn.Sequential(
            nn.Linear(256, 512),
        )
        
        self.dense_decode = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.voice_decoder1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.voice_decoder2 = nn.Sequential(
            nn.Linear(128, input_size),
        )

        self.classifier = nn.Linear(512, n_classes)
    
    def forward(self, x, mode='inference'):
        feat1 = self.voice_encoder1(x)
        feat2 = self.voice_encoder2(feat1).permute(0, 2, 1)
        feature1 = self.dense_encode(feat2)
        feature2 = feature1.permute(0, 2, 1).mean(-1)
        feature2 = self.classifier(feature2)
        
        # return self.voice_classifier(feature)  # Fully connected layer
        if mode == 'train':
            feature1 = self.dense_decode(feature1)
            feature1 = self.voice_decoder1(feature1)
            feature1 = self.voice_decoder2(feature1).permute(0, 2, 1)
        
            return feature1, feature2
        else:
            return feature1.permute(0, 2, 1).mean(-1), feature2
        
        

    def set_params(self, input_size=40, n_classes=2, sample_rate=16000):
        # 파라미터 설정
        self.target_sample_rate = sample_rate  # 타겟 샘플 속도
        self.n_mfcc = input_size  # MFCC의 수
        self.n_classes = n_classes  # 구별해야 할 클래스 수 (사이렌 소리와 일반 소리)
        self.N2C = {0: 'heetae', 1: 'sangbeom', 2: 'soyoung', 3: 'minseop', 4: 'yunjung', 5: 'unknown'}
        self.person_features = {}
        self.registered_people = {'heetae':0, 'sangbeom':1, 'soyoung':2, 'minseop':3, 'yunjung':4}


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

    def predict(self, path, mode='train'):
        waveform = self.load_audio(path)
        mfcc_features = self.extract_features(waveform[0])
        inputs = mfcc_features.unsqueeze(0)
        inputs = self.z_score_normalize(inputs)
        output = self.forward(inputs.cuda(), mode)
        if mode=='train':
            result = output.argmax(-1).item()
            return result
        else:
            return output


    def voicecheck(self):

            print(f"Listening for 3 seconds...")
            audio_data = sd.rec(int(3 * self.target_sample_rate), samplerate=self.target_sample_rate, channels=1, dtype='int16')
            sd.wait()  # Wait for the recording to finish
            write('voice.wav', self.target_sample_rate, audio_data)

            features1 = {i:[] for i in range(5)}
            features2 = {i:[] for i in range(5)}

            data = [os.path.join('Person', per) for per in os.listdir('Person')]

            for d in data:
                feat1, feat2 = self.predict(d, 'inference')
                if 'heetae' in d:
                    features1[0].append(feat1)
                    features2[0].append(feat2)
                elif 'sangbeom' in d:
                    features1[1].append(feat1)
                    features2[1].append(feat2)
                elif 'soyoung' in d:
                    features1[2].append(feat1)
                    features2[2].append(feat2)
                elif 'minseop' in d:
                    features1[3].append(feat1)
                    features2[3].append(feat2)
                elif 'yunjung' in d:
                    features1[4].append(feat1)
                    features2[4].append(feat2)

            file = 'voice.wav'
            feature1, feature2 = self.predict(file, 'inference')
            wave = self.load_audio(file)
            wave = self.extract_features(wave).cuda()
            wave = self.z_score_normalize(wave)


            similarity1 = []
            similarity2 = []
            loss1 = []
            loss2 = []

            for key, val in features1.items():
                similarity1.append(torch.cosine_similarity(feature1[0], torch.cat(val)).mean(-1).max().unsqueeze(0))
                loss1.append(((feature1 - torch.cat(val)).pow(2)).mean(-1).min().unsqueeze(0))
            for key, val in features2.items():
                similarity2.append(torch.cosine_similarity(feature2[0], torch.cat(val)).mean(-1).max().unsqueeze(0))
                loss2.append(((feature2 - torch.cat(val)).pow(2)).mean(-1).min().unsqueeze(0))


            similarity1 = torch.cat(similarity1)
            similarity2 = torch.cat(similarity2)
            loss1 = torch.cat(loss1)
            loss2 = torch.cat(loss2)
            predict = self.forward(wave, 'train')[-1].argmax()
            if predict == similarity1.argmax() == similarity2.argmax():
                print('Owner')
            else:
                print('Unknown')
            predict, similarity1.argmax(), similarity2.argmax()


    # 모델 훈련
    def train_model(self, data, lr=0.001, epochs=15, mode='train'):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        pbar = tqdm(range(epochs))
        self.train()
        for epoch in pbar:
            inputs, label = data[0], data[1]
            output1, output2 = self.forward(inputs, mode)  # (1, n_mfcc, time_steps)
            encoding_loss1 = (output1 - inputs).pow(2).mean()
            encoding_loss2 = (1-torch.cosine_similarity(output1, inputs, dim=-1)).mean()
            encoding_loss = encoding_loss1  + encoding_loss2
            cls_loss = criterion(output2, label)
            loss = cls_loss + encoding_loss
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.6f}, loss1: {cls_loss:.6f}, loss2: {encoding_loss:.6f}")


if __name__ == '__main__':

    model = VoiceRecognizer(input_size=120, n_classes=6).cuda()
    model.load_state_dict(torch.load('Voice_Check.pt'))
    model.voicecheck()