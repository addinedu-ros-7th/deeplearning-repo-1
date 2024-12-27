# 차량 서비스 ###

> 부제 : Segmentation 및 Object Detection  
> 팀명 : CASS(CASS : CAr ServiceS , 차량의 모든 서비스)

## 1. 프로젝트 개요

### 1.1. 소개
(뉴스 첨부)


### 1.2. 프로젝트 목표(안전과 편의)
* 차선 유지
* 신호 준수
* 응급 상황 대응
* 장애물 실시간 대응
* 졸음 인식 대응
* 음성 인식 서비스

---   






## 2. 시스템 구성   
### 2.1. system requirements
|분류|기능|설명|우선순위|
|---|---|---|---|
|수동 제어|전진|사용자가 앞으로 키 누를 시 전진|1|
|||음성 인식을 활용하여 전진 명령 수행|4|
||후진|사용자가 뒤로 키 누를 시 후진|3|
|||음성 인식을 활용하여 후진 명령 수행|4|
||정지|사용자가 정지 키 누를 시 제자리 정지|1|
|||음성 인식을 활용하여 정지 명령 수행|4|
|자동제어|차선 유지|중앙 차선과 바깥 차선을 인식하여 차선을 지키며 주행|1|
|||사용자가 입력한 방향으로 경로 설정하여 주행|2|
||객체 인식|정지선, 횡단보도, 신호등(빨간색, 초록색), 장애물 인식|1|
||신호 대응|신호가 빨간 불일 때 정지선 앞에 정지|1|
|||신호가 초록불일 때 정지선 통과 및 출발|1|
||동적 장애물 대응|사물의 이동 경로를 파악하여 적절한 회피 동작 수행|3|
|||사물이 지나갈 때까지 기다린 후 출발|2|
||정적 장애물 대응|정적 장애물 인식 시 중앙선 침범하여 회피 주행|2|
|보안|얼굴 인식|1차 보안을 위한 사용자 얼굴 인식|1|
||음성 인식|2차 보안을 위한 사용자 음성 인식|4|
|응급상황|졸음 운전 대응|졸음 인식 시 졸음 경고 및 부저 작동|1|
|||졸음 인식 시 반복 제동 수행|2|
|||사람이 일어나지 않는다면 갓길 주차|4|
||응급 차량 대응|교차로에서 응급차량이 지나갈 시 정지|4|
|||동차선에서 응급차량 지나갈 시 ***갓길**로 회피|4|   

*갓길-후면카메라가 있다면 주행방향을 선택할 수 있겠지만 일차적으로 갓길로 제한을 둠.

### 2.2. System Architecture
<img src="https://github.com/user-attachments/assets/8aa1809b-4dad-46a1-91dc-6f13010a73ef" width="500"/><br/>

### 2.3. State Diagram

* State Diagram - 전체
  
<img src="https://github.com/user-attachments/assets/c42af502-7edc-47f0-b7da-32a74b11be84" width="500"/><br/>   

* State Diagram - 주행
  
<img src="https://github.com/user-attachments/assets/c40b1d02-db2b-49bd-8a0c-1aab41642fde" width="500"/><br/>

### 2.4. Sequence Diagram
* Sequence Diagram - 회피 주행
  
<img src="https://github.com/user-attachments/assets/105eb515-0f8a-48f5-934e-c89579f577ed" width="700"/><br/>



---   

## 3. 개발 일정 및 역할 분담

### 3.1. 팀 구성원별 담당 사항 &nbsp;&nbsp;&nbsp;&nbsp;
|구분|이름|업무|
|---|---|---|
|팀장|이상범|갓길 주차 알고리즘, 차선 유지 알고리즘, 졸음 인식 알고리즘, 음성 인증 알고리즘, 응급 차량 인식 알고리즘, TCP/IP 통신 연결|
|팀원|윤희태|음성 대화 알고리즘 개발, 주행 환경 인식 모델 개발, 주행 환경 구성|
|팀원|김소영|사용자 얼굴인식 알고리즘, GUI 구현, TCP/IP 통신 구축, Confluence / Jira Sprint 관리, Git 버전/태그 관리, DB 구축|
|팀원|윤민섭|차량 동작 알고리즘 개발, 차량 부품 조립, TCP/IP 통신 연결, 주행 환경 구성|
|팀원|장윤정|졸음 인식 알고리즘 개발, GUI 구성, 주행 환경 구성, Git 관리|

### 3.2. 일정 관리
* Jira 타임라인
  
<img src="https://github.com/user-attachments/assets/411eb735-5965-4a57-b3de-358f6062e63b" width="500"/><br/>

* Jira Sprint
  
<img src="https://github.com/user-attachments/assets/14f37285-149b-45f9-beda-1ad71d52fc2f" width="500"/><br/>


### 3.3. 활용 기술
|구분|상세|
|------|----------------------|
|개발환경|<img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white" style="width: 67px; height: 20px;"> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white">|
|딥러닝 및 영상처리| <img src="https://github.com/user-attachments/assets/5f8d52f1-1b12-4075-a59d-a641c01ad558" style="width: 67px; height: 20px;"> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white" /> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white" style="width: 67px; height: 20px;">
|데이터베이스|<img src="https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=MySQL&logoColor=white">|
|GUI| <img src="https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=Qt&logoColor=white" style="width: 67px; height: 20px;">
|통신|<img src="https://img.shields.io/badge/SocKet-C93CD7?style=flat&logo=SocKet&logoColor=white" />|
|하드웨어|<img src="https://img.shields.io/badge/ARDUINO UNO-00878F?style=for-the-badge&logo=Arduino&logoColor=white" style="width: 97px; height: 20px;">
|형상관리 및 협업|<img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=white" style="width: 67px; height: 20px;"/> <img src="https://img.shields.io/badge/Confluence-172B4D?style=flat&logo=Confluence&logoColor=white" style="width: 87px; height: 20px;"><img src="https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=Jira&logoColor=white" style="width: 87px; height: 20px;">|

---   

## 4. 결과

### 4.1. 기능별 영상

* 차선유지
  
<img src="https://github.com/user-attachments/assets/79043081-8f52-4954-9137-3491db77f19a" height="280"/>
<img src="https://github.com/user-attachments/assets/f46b42c1-5772-4d4b-8c73-25289aced102" width="500"/><br/>
  
* 신호준수
  
<img src="https://github.com/user-attachments/assets/6d73febc-3896-4611-9458-3fbbd2853b9a" width="500"/><br/>
<img src="" width="500"/><br/>

* 동적 장애물 정지
  
<img src="https://github.com/user-attachments/assets/7dd85076-4389-4137-bea0-d1840de656eb" width="500"/><br/>
<img src="" width="500"/><br/>

* 정적 장애물 회피
  
<img src="https://github.com/user-attachments/assets/ef43e349-135c-48bb-8ec5-3ec66727ff91" width="500"/><br/>
<img src="" width="500"/><br/>

* 졸음인식 기상제동

<img src="https://github.com/user-attachments/assets/03090a1e-2db6-4366-a2ca-e3e548064084" height="285"/>
<img src="https://github.com/user-attachments/assets/23af594c-a4fd-45d7-a0ee-383c7f713ea4" width="500"/><br/>

* 응급상황 갓길주차

<img src="https://github.com/user-attachments/assets/ba6911a7-a160-469f-8f29-dc3214e3db5c" height="320"/> 
<img src="https://github.com/user-attachments/assets/1ee69d80-63f6-4fb7-8030-370c57b30a64" width="500"/><br/>

* 얼굴인식 / 음성인식 보안인증

<img src="https://github.com/user-attachments/assets/fdaf5a00-fa2c-4c79-8c05-9be700800443" width="500"/>

https://github.com/user-attachments/assets/f60c1f17-6808-447d-baa6-fd2412580095

* 음성명령

https://github.com/user-attachments/assets/d443c966-e62f-4cb3-9057-f16dc9be2318

  
### 4.2. 상세 구현 결과
|분류|상세 분류|기능|결과|
|---|---|---|---|
|수동 제어|버튼 조작|W 버튼 누를 시 전진|PASS|
|||S 버튼 누를 시 전진|PASS|
|||D 버튼 누를 시 전진|PASS|
||음성 명령|시동 명령에 시동 걸기|PASS|
|||출발 명령에 차량 출발|PASS|
|||시동 종료 명령에 시동 종료|PASS|
|||정지 명령에 정차|PASS|
|||차량의 현재 상태와 중복된 명령 구분|PASS|
|자동 제어|차선 유지|차선 중앙을 따라 주행|PASS|
|||교차로에서 탑승자가 지정한 진행 방향대로 주행|PASS|
||객체 대응|적신호 시 정지선 앞에서 정지|PASS|
|||청신호 시 정지선 통과 및 주행|PASS|
|||동적 장애물의 이동에 따른 회피 동작 수행|FAIL|
|||동적 장애물이 인식되면 정지|PASS|
|||동적 장애물이 사라지면 출발|PASS|
|||정적 장애물 인식 시 회피 주행|PASS|
|보안|얼굴 인증|사용자 얼굴 인식 후 차량 통제권 부여|PASS|
||음성 인증|사용자 음성 인식 후 차량 통제권 부여|PASS|
|응급 상황|차주 졸음 의심|반복 제동 수행|PASS|
|||무반응 시 갓길 주차|PASS|
||응급 차량 인식|소리 인식 시 갓길로 회피|PASS|



### 4.3. 구현 중 발생한 이슈 및 해결 과정

#### 주행 관련 AI ▽ ####
* Lane Segmentation을 통해 차선의 중심을 잡을 경우 중심점이 많이 튀는 문제가 발생함.
  >Road Segmentation 정보를 학습하여 안정적으로 중심점을 잡을 수 있도록 개선함.

* Segmentation Points의 평균값을 취할 경우, 면적의 중점이 아닌 포인트가 몰려있는 부분으로 찾는 경우가 발생.
  >각각의 point에 대한 면적을 계산하여 Segmentation 면적에 대한 중심점을 찾음.

* 주행 환경 인식 모델 중 객체와 카메라 사이의 거리를 계산하는 과정에서 해상도 변경으로 거리 값이 부정확해짐.
  >기존 거리 측정 알고리즘은 640x480 해상도의 프레임에서 인식된 객체의 바운딩 박스 픽셀 값을 사용하게 되는데, 실제 서비스 화면에서는 960x640 해상도를 사용하므로, 변경된 해상도에서 픽셀값을 사용하는게 아닌, 원래 해상도에서 거리까지 모두 구한 후의 최종 프레임을 새로운 해상도로 늘리는 방법을 택함.

#### 음성 AI ▽ ####

* 현재의 상태나 이전의 대화 내용을 기억하지 못함.
  >이전에 챗봇을 통해 시동을 건 상태에서 시동을 걸어달라 혹은 정차한 상태에서 정차해 달라는 등 현재 상태와 중복된 명령을 하달하는 경우, 단순히 이를 다시 실행하는 답변을 출력하게 됨. 그래서 챗봇 상에 차량의 상태를 저장해 놓는 알고리즘을 구현해 현재 상태와 중복된 명령을 받을 경우 이를 알아차리고 구분해내는 답변을 출력하도록 함.


* 학습에 사용되지 않은 사람의 음성을 잘 구별해내지 못함 (모르는 사람을 모른다고 못함).
  >Classification output, Cosine Similarity, Encoded Voice 정보 등 추가적인 정보를 이용하여 출력된 값이 모두 일치하고, loss의 거리가 충분히 가까울 때 등록된 사람으로 판단하도록 만들어, 음성 보안 시스템을 강화시킴.

  >---      
  > 사용자 목소리의 특징을 잘 분별 해내기 위해서 단순 Classification 모델에서 Auto Encoder 모델로 바꿔 학습시킴.
  > 학습을 할 때는 Encoded Voice 정보를 이용하여 Cross Entropy Loss(Classification)를 통해 학습 시켰고, Decoder의 출력값은 
  > Mean Squared Error와 Cosine Similarity를 사용한 Loss를 통해 사용자의 특징을 더욱 잘 분별할 수 있도록 학습시킴. 
  > * CE Loss + MSE Loss + CS Loss → 분별력 있는 학습 진행.
  

#### 하드웨어 ▽ ####
* 전선의 접촉상태, 세트장 환경, 차량 무게, 차량의 무게중심 등 여러가지 요인으로 원하는 동작이 이루어지지 않음.
  >전선을 테이프로 고정, 세트장 및 차량에 자갈을 얹어 평평한 지면에 완벽히 접지를 이룰 수 있도록 하드웨어를 보완함.

* 전원 부족으로 WIFI 연결이 끊기거나, WIFI는 정상인데 단말기가 접속하지 못하는 문제가 발생함.
  >5V 3A 보조배터리를 사용하여 안정적인 전원공급을 시도했고, 문제 없는 시설 WIFI를 찾아 사용함.

#### 통신 ▽ ####
* 동기식 TCP 소켓 통신으로 ESP32에 주행 명령을 보낼 경우 응답 속도에 따라 전체 프로세스가 버벅이게 됨.
  >비동기식 TCP 통신을 사용해 응답이 느리더라도 계속해서 명령을 전달할 수 있도록 통신 속도를 개선함.

* 하나의 프로세스에서 과도한 스레드를 사용하다보니 인공지능의 성능이 저하됨.
  >스레드로 여러가지 인공지능 모델을 분리 구동하는 게 아닌, 서버를 분리하여 TCP 통신을 통해 결과를 받아와 메인 서버에 업데이트 하는 방식으로 인식 성능을 개선함.


### 4.4. 개선 사항

* 프로젝트 시작 전 기술의 실현가능성을 파악하지 않은 채 기능리스트를 작성하여 이후 서버와 차량 간 결합에 어려움이 있었음.   
▶ 소프트웨어와 하드웨어의 기술적 한도를 충분히 파악한 후 구현에 착수해야 함.

