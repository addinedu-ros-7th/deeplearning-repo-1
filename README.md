### 차량 서비스 ###

> 부제 : Segmentation 및 Object Detection  
> 팀명 : CASS(CASS : CAr ServiceS , 차량의 모든 서비스)

## 0. 최종 시연 영상

## 1. 프로젝트 개요

### 1.1. 활용기술

|구분|상세|
|------|----------------------|
|개발환경|<img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white" style="width: 67px; height: 20px;"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white">|
|딥러닝 및 영상처리| <img src="https://github.com/user-attachments/assets/5f8d52f1-1b12-4075-a59d-a641c01ad558" style="width: 67px; height: 20px;"> <img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" /> <img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" />|
|데이터베이스|<img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" />|
|GUI|<img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" />|
|통신|<img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" />|
|하드웨어|<img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" />|
|형상관리 및 협업|<img src="https://img.shields.io/badge/GitHub-1976D2?style=flat-square&logo=GitHub&logoColor=white"/> <img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" /> <img src="https://img.shields.io/badge/Ultralytics-1976D2?style=flat&logo=Ultralytics&logoColor=white" />|

### 1.2. 프로젝트 목표
- <b>차선 인식을 통한 주행 경로 유지 제어(Segmentation)</b><br/>
  <ul style="list-style: circle;">
   <li>주행 차선 인식 : 주행 중 양측 차선(외곽선, 중앙선, 교차로) 정보 실시간 인지</li>
   <li>주행 경로 판단 : 차선 정보를 활용한 이동 위치 판단 </li>
  </ul>
- 교통 객체 인식을 통한 상황 판단(Object Detection) <br/>
  <ul style="list-style: circle;">
   <li>교통 객체 인식 : 주행 중 신호등, 방향 지시, 정지선/횡단보도, 차량, 보행자 객체 실시간 인지</li>
   <li>차량 제어 : 교통 상황별 차량의 주행, 서행, 정지 제어</li>
  </ul>



## 2. 시스템 구성
### 2.1. 기능 리스트
### 2.2. 시스템 구성도
![image_720](https://github.com/user-attachments/assets/705c46b3-c18c-4498-9cf0-a5d17c418748)

### 2.3. 차선 및 객체 인식 시퀀스
<img src="https://github.com/user-attachments/assets/b2c9e22a-9c23-4ee7-a101-cc7881217aa8" alt="image" width="720" height="482">

### 2.4. 전체 시나리오 시퀀스
<img src="https://github.com/user-attachments/assets/cc867796-b1eb-4910-a5ea-62127b9c0281" alt="image" width="720" height="482">

## 3. 개발 일정 및 역할 분담
### 3.1. 이슈별 일정 관리
### 3.2. 팀 구성원별 담당 사항

## 4. 결과
### 4.1. 상세 구현 결과
### 4.2. 구현 중 발생한 이슈 및 해결 과정
### 4.3. 개선 사항
