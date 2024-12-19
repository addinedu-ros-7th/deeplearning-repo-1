#include <WiFi.h>

// 환경 세팅
const char* ssid = "AIE_509_2.4G";  // WIFI
const char* password = "addinedu_class1";   // WIFI 비밀번호

const int RED1 = 13; // 후미등 1
const int RED2 = 4; // 후미등 2
const int BLUE1 = 18; // 차주 인식등

const int motor_R_forward = 21;  // 오른쪽 정방향 IN1
const int motor_R_reverse = 19;  // 오른쪽 역박향 IN2

const int motor_L_forward = 26;  // 왼쪽 정방향 IN3
const int motor_L_reverse = 27;  // 왼쪽 역박향 IN4


// 정지 및 직후진, L1~3, R1~3 모드 저장용 변수
int select_num;
int steering_mode;

// 차량 기본 속도 세팅
const int speed = 100;
const int basic_left_speed = speed + 40;
const int basic_right_speed = speed;

// 현재 좌우측륜 속력 저장용 변수
int curr_right_speed;
int curr_left_speed;

// 좌우측륜 가감속 구조체
struct Delta{
  int l_delta;
  int r_delta;
};
Delta d = {0, 0}; // 기본 가감속
Delta brake = {0, 0}; // Siren 전용 감속 변수

// 싸이렌 응급상황 대응변수
bool Siren = false;
bool prev_siren = true;

// LED millis() 세팅
unsigned long LED_start_time = 0;
const unsigned long LED_DELAY_TIME = 5000;

// BLUE LED 변수
bool BLUE_ON = false;

// 탑승자 무반응 비상상황 대응변수
bool Emergency_Signal = false;
int Emergency_cnt = 0;

// 주행상황 플래그 변수 (정지 및 직후진)
bool StopFlag = true;
bool BackFlag = false;
bool isReady = false;

// 속도 출력용 변수
int show_speed = 0;

// 차선 탐색 후 직진 변환용 변수
int prev_steer_status;
int curr_steer_status;
bool SearchForLines = false;
bool isRight = false;


WiFiServer server(500);

// BLUE LED 동작 함수
void LED_startTimer()
  {
    LED_start_time = millis();
  }

bool LED_checkTimer()
{
  unsigned long now = millis();
  if (LED_start_time == 0)
  {
    return false;
  }
  else if((now - LED_start_time) <= LED_DELAY_TIME)
  {
    return true;
  }
  
  return false;
}

// 직진
void GO_FORWARD()
{
  curr_left_speed = basic_left_speed + d.l_delta - brake.l_delta;
  curr_right_speed = basic_right_speed + d.r_delta - brake.r_delta;

  analogWrite(motor_L_forward, curr_left_speed);
  analogWrite(motor_L_reverse, 0);

  analogWrite(motor_R_forward, curr_right_speed);
  analogWrite(motor_R_reverse, 0);

}

// 후진
void GO_BACK()
{
  curr_left_speed = basic_left_speed + d.l_delta-brake.l_delta;
  curr_right_speed = basic_right_speed + d.r_delta-brake.r_delta;
  
  analogWrite(motor_R_forward, 0);
  analogWrite(motor_R_reverse, curr_right_speed);

  analogWrite(motor_L_forward, 0);
  analogWrite(motor_L_reverse, curr_left_speed);
}

// 정지
void STOP()
{
  curr_left_speed = basic_left_speed + d.l_delta;
  curr_right_speed = basic_right_speed + d.r_delta;

  analogWrite(motor_R_forward, curr_right_speed);
  analogWrite(motor_R_reverse, curr_right_speed);

  analogWrite(motor_L_forward, curr_left_speed);
  analogWrite(motor_L_reverse, curr_left_speed);
}

// 차선 탐색용 지그재그 주행
void ZIGZAG(bool isRight)
{
  curr_left_speed = basic_left_speed + d.l_delta;
  curr_right_speed = basic_right_speed + d.r_delta;

  if (isRight)
  {
    analogWrite(motor_R_forward, curr_right_speed);
    analogWrite(motor_R_reverse, 0);

    analogWrite(motor_L_forward, 0);
    analogWrite(motor_L_reverse, curr_left_speed);
  }
  else
  {
    analogWrite(motor_R_forward, 0);
    analogWrite(motor_R_reverse, curr_right_speed);

    analogWrite(motor_L_forward, curr_left_speed);
    analogWrite(motor_L_reverse, 0);
  }
}

// 탑승자 무반응 비상상황
void Emergency()
{
  if (Emergency_cnt < 10)
  {
    d = {-basic_left_speed, -basic_right_speed};
    STOP();
  }
  else if (Emergency_cnt < 20)
  {
    d = {110, 150};
    GO_FORWARD();
  }
  else if (Emergency_cnt < 30)
  {
    d = {-basic_left_speed, -basic_right_speed};
    STOP();
  }
  else if (Emergency_cnt < 40)
  {
    d = {110, 150};
    GO_FORWARD();
  }
  else if (Emergency_cnt < 50)
  {
    d = {-basic_left_speed, -basic_right_speed};
    STOP();
  }
  else if (Emergency_cnt < 60)
  {
    d = {110, 150};
    GO_FORWARD();
  }
  else if (Emergency_cnt < 75)
  {
    d = {-basic_left_speed, -basic_right_speed};
    STOP();
    StopFlag = true;
  }

  Emergency_cnt ++;

  if (Emergency_cnt >= 100)
  {
    Emergency_Signal = false;
    Emergency_cnt = 0;
  }
}

void SearchForLine()
{
  static int cnt = 0;
  cnt ++;
  if (cnt <= 60)
  {
    if (cnt >= 30)
    {
      ZIGZAG(true);
    }
    else
    {
      ZIGZAG(false);
    }
  }
  else
  {
    cnt = 0;
  }
}

void setup() {
  Serial.begin(115200);  // 시리얼 모니터 시작

  pinMode(RED1, OUTPUT);
  pinMode(RED2, OUTPUT);
  pinMode(BLUE1, OUTPUT);

  pinMode(motor_R_forward, OUTPUT);
  pinMode(motor_R_reverse, OUTPUT);

  pinMode(motor_L_forward, OUTPUT);
  pinMode(motor_L_reverse, OUTPUT);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) 
  {
    delay(1000);
    Serial.println("와이파이 연결 중...");
  }

  Serial.println("와이파이 연결완료");
  Serial.print("IP 주소: ");
  Serial.println(WiFi.localIP());  // ESP32의 IP 주소 출력

  server.begin();  // 웹 서버 시작
  int data;
}

void loop() {
  WiFiClient client = server.available();

  while (client.connected())
  {
    show_speed ++;
    client.setTimeout(20);
    String data = client.readStringUntil('\n'); // 줄바꿈("\n") 기준으로 데이터 읽기 
    data.trim(); // 데이터 앞뒤 공백 제거

    // 서버로부터 명령어 수신
    switch (data.toInt()) {

      case 11: // 연결 확인
        client.print("connected\n");
        Serial.println("클라이언트 연결 성공\n");
        break;
      
      case 21: // 차주 인식 + 동작 가능
        client.print("welcome soyoung\n");
        select_num = 0;
        isReady = true;
        StopFlag = true;
        BLUE_ON = true;
        LED_startTimer(); 
        break;

      case 22: // 운행 종료
        if (StopFlag)
        {
          client.print("Have a nice day.\n");
          select_num = 0;
          isReady = false;
          // StopFlag = true;
          BLUE_ON = true;
          LED_startTimer();
        }
        break;

      case 31: // 직진
        BackFlag = false;
        StopFlag = false;
        SearchForLines = false;
        select_num = 1;
        break;

      case 32: // 정지
        select_num = 0;
        SearchForLines = false;
        BackFlag = false;
        StopFlag = true;
        break;

      case 33: // 후진
        BackFlag = true;
        select_num = 1;
        StopFlag = false;
        SearchForLines = false;
        break;

      case 34: // 가속
        select_num = 8;
        StopFlag = false;
        SearchForLines = false;
        break;
      
      case 41: // L1
        StopFlag = false;
        select_num = 2;
        SearchForLines = false;    
        break;
      
      case 42: // L2
        StopFlag = false;
        SearchForLines = false;
        select_num = 3;
        break;

      case 43: // L3
        StopFlag = false;
        SearchForLines = false;
        select_num = 4;
        break;

      case 51: // R1
        StopFlag = false;
        SearchForLines = false;
        select_num = 5;
        break;

      case 52: // R2
        StopFlag = false;
        SearchForLines = false;
        select_num = 6;
        break;
      
      case 53: // R3
        StopFlag = false;
        SearchForLines = false;
        select_num = 7;        
        break;

      case 77: // 차선 탐색
        StopFlag = false;
        select_num = 9;
        break;

      case 88: // 응급상황 갓길주차
        Siren = true;
        SearchForLines = false;
        break;

      case 99: // 졸음 운전 대응
        StopFlag = false;
        SearchForLines = false;
        Emergency_Signal = true;
        break;
    }

    prev_steer_status = curr_steer_status;

    //LED 작동함수
    if (BLUE_ON && LED_checkTimer())
    {
      digitalWrite(BLUE1, HIGH);
    }
    else
    {
      BLUE_ON = false;
      digitalWrite(BLUE1, LOW);
    }

    // Serial.print("select_num: ");
    // Serial.println(select_num);
    // Serial.print("steering_mode: ");
    // Serial.println(steering_mode);
    
    if (isReady)
    {
      steering_mode = select_num;
      // 명령별 조향모드 설정 (기본 가감속 구조체 변수 'd' 사용)
      switch (steering_mode) {
        case 0: // 정지
          d = {-basic_left_speed, -basic_right_speed};

        case 1: // 직진
          d = {0, 0};
          break;

        case 2: // L1
          d = {-40, 90};
          break;

        case 3: // L2
          d = {-45, 110};
          break;

        case 4: // L3
          d = {-50, 155};
          break;
        
        case 5: // R1
          d = {60, -20};
          break;

        case 6: // R2
          d = {85, -20};
          break;

        case 7: // R3
          d = {115, -20};
          break;

        case 8: // ACCEL
          d = {40, 40};
          break;

        case 9: // 차선 탐색
          d = {20, 40};
          SearchForLines = true;
          break;
      }

      curr_steer_status = steering_mode;

      if (curr_steer_status != 9 && prev_steer_status == 9)
      {
        BackFlag = false;
      }

      if(Siren && prev_siren) // 응급상황 감속
      {
        if (steering_mode > 0) // 정지 제외 나머지 모드에서만 작동
        {
          // Siren 전용 감속 구조체 변수 'brake' 사용
          brake.l_delta+=1;
          brake.r_delta+=1;
          delay(15);
          if (StopFlag)
          {
            STOP();
          }
          else
          {
            if(SearchForLines)
            {
              SearchForLine();
            }
            else
            {
              if(BackFlag)
              {
                GO_BACK();
              }
              else
              {
                GO_FORWARD();
              }
            }
          }
        }
        if (curr_right_speed <= 0 || curr_left_speed <= 0) // 감속 중에 현재속도 음수될 시 정차
        {
          d = {-basic_left_speed, -basic_right_speed};
          brake = {0, 0};
          StopFlag = true;
          Siren = false;
          prev_siren = false;
          STOP();
        }

      } 
      else // 응급상황 아닐 시 정상주행
      {
        prev_siren = true;
        if (StopFlag) // 정지 플래그 true - 정지
        {
          SearchForLines = false;
          STOP();
          digitalWrite(RED1, HIGH);
          digitalWrite(RED2, HIGH);
        }
        else // 정지 플래그 false - 직진 혹은 후진
        {
          digitalWrite(RED1, LOW);
          digitalWrite(RED2, LOW);
          if(SearchForLines)
          {
            SearchForLine();
          }
          else
          {
            SearchForLines = false;
            if (BackFlag) // 후진 플래그 true - 후진
            {
              GO_BACK();
            }
            else // 후진 플래그 false - 직진
            {
              GO_FORWARD();
            }
          }
        } 
      }

      //비상 제동
      if (Emergency_Signal)
      {
        Emergency();
      }
      else
      {
        Emergency_Signal = false;
        continue;
      }
    }
  }
}