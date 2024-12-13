#include <WiFi.h>

// const char* ssid = "Addinedu_509_office_5G";  // Wi-Fi SSID777
// const char* password = "Addinedu8565!";        // Wi-Fi 비밀번호

const char* ssid = "AIE_509_2.4G";  // WIFI
const char* password = "addinedu_class1";   // WIFI 비밀번호

const int RED1 = 13;
const int RED2 = 4;
const int BLUE1 = 18;

const int speed = 100;
int steering_mode;

const int basic_left_speed = speed + 40;//+40;
const int basic_right_speed = speed;

int right_speed;
int left_speed;

bool acc_ON; // 가속 보정
int acc_cnt = 0; //가속 보정용 count 변수
static int curr_status = 0;
static int prev_status = 0;

// LED, 부저 millis() 세팅
unsigned long LED_start_time = 0;
const unsigned long LED_DELAY_TIME = 5000;


bool BLUE_ON = false;

const int motor_R_forward = 21;  // 오른쪽 정방향 IN1
const int motor_R_reverse = 19;  // 오른쪽 역박향 IN2

const int motor_L_forward = 26;  // 왼쪽 정방향 IN3
const int motor_L_reverse = 27;  // 왼쪽 역박향 IN4

bool Emergency_Signal = false;
int Emergency_cnt = 0;

bool StopFlag = true;
bool BackFlag = false;

// 속도 출력용 변수
int show_speed = 0;

WiFiServer server(8080);


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

// void ACCEL()
// {
//   if (acc_cnt >= 20)
//   {
//     steering_mode = 1;
//     left_speed = basic_left_speed;
//     right_speed = basic_right_speed;
//     acc_ON = false;
//     acc_cnt = 0;
//   }
//   else
//   {
//     left_speed = basic_left_speed + 40;
//     right_speed = basic_right_speed + 40;
//   }
// }
void GO_FORWARD()
{
  analogWrite(motor_L_forward, left_speed);
  analogWrite(motor_L_reverse, 0);

  analogWrite(motor_R_forward, right_speed);
  analogWrite(motor_R_reverse, 0);
}

void GO_BACK()
{
  analogWrite(motor_R_forward, 0);
  analogWrite(motor_R_reverse, right_speed);

  analogWrite(motor_L_forward, 0);
  analogWrite(motor_L_reverse, left_speed);
}

void STOP()
{
  analogWrite(motor_R_forward, right_speed);
  analogWrite(motor_R_reverse, right_speed);

  analogWrite(motor_L_forward, left_speed);
  analogWrite(motor_L_reverse, left_speed);
}

void Emergency()
{
  if (Emergency_cnt < 10)
  {
    STOP();
  }
  else if (Emergency_cnt < 25)
  {
    left_speed = 250;
    right_speed = 250;
    GO_FORWARD();
  }
  else if (Emergency_cnt < 35)
  {
    STOP();
  }
  else if (Emergency_cnt < 55)
  {
    left_speed = 250;
    right_speed = 250;
    GO_FORWARD();
  }
  else if (Emergency_cnt < 65)
  {
    STOP();
  }
  else if (Emergency_cnt < 85)
  {
    left_speed = 250;
    right_speed = 250;
    GO_FORWARD();
  }
  else if (Emergency_cnt < 95)
  {
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

void setup() {
  Serial.begin(115200);  // 시리얼 모니터 시작
  
  pinMode(RED1, OUTPUT);
  pinMode(RED2, OUTPUT);
  pinMode(BLUE1, OUTPUT);

  // pinMode(motor_R, OUTPUT);
  pinMode(motor_R_forward, OUTPUT);
  pinMode(motor_R_reverse, OUTPUT);

  // pinMode(motor_L, OUTPUT);
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
  right_speed = basic_right_speed;
  left_speed = basic_left_speed;  
}

void loop() {
  WiFiClient client = server.available();

  while (client.connected())
  {
    show_speed ++;
    client.setTimeout(20);
    String data = client.readStringUntil('\n'); // 줄바꿈("\n") 기준으로 데이터 읽기 
    data.trim(); // 데이터 앞뒤 공백 제거

    switch (data.toInt()) {

      case 11: // 연결 확인
        client.println("connected");
        Serial.println("클라이언트 연결 성공");
        break;
      
      case 21: // 차주 인식
        client.println("welcome soyoung");
        BLUE_ON = true;
        LED_startTimer(); 
        break;

      case 31: // 직진
        // client.println("drive");
        if (StopFlag)
        {
          acc_ON = true;
        }
          BackFlag = false;
          StopFlag = false;
          steering_mode = 1;
        break;

      case 32: // 정지
        // client.println("stop");
        steering_mode = 0;
        BackFlag = false;
        StopFlag = true;
        //BUZZER_ON = true;
        // BUZZER_startTimer();
        break;

      case 33: // 후진
        // client.println("reverse");
        if (StopFlag)
        {
          BackFlag = true;
          acc_ON = true;
        }
        steering_mode = 1;
        StopFlag = false;
        break;

      case 34:
        client.println("accel");
        steering_mode = 8;
        StopFlag = false;
        break;
      
      case 41: // L1
        // client.println("L1");
        StopFlag = false;
        steering_mode = 2;    
        break;
      
      case 42: // L2
        // client.println("L2");
        StopFlag = false;
        steering_mode = 3;

        break;

      case 43: // L3
        // client.println("L3");
        StopFlag = false;
        steering_mode = 4;

        break;

      case 51: // R1
        // client.println("R1");
        StopFlag = false;
        steering_mode = 5;        
        
        break;

      case 52: // R2
        // client.println("R2");
        StopFlag = false;
        steering_mode = 6;
        break;
      
      case 53: // R3
        // client.println("R3");
        StopFlag = false;
        steering_mode = 7;        
        break;

      case 99: // 비상 제동
        // client.println("Emergency Breaking");
        StopFlag = false;
        Emergency_Signal = true;
        break;
    }

    switch (steering_mode) {
      case 0: // 정지
        left_speed = 0;
        right_speed = 0;

      case 1: // 직진
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;
        break;

      case 2: // L1
        right_speed = basic_right_speed + 80;
        left_speed = basic_left_speed - 35;
        break;

      case 3: // L2
        right_speed = basic_right_speed + 110;
        left_speed = basic_left_speed - 35;
        break;

      case 4: // L3
        right_speed = basic_right_speed + 145;
        left_speed = basic_left_speed - 30;
        break;
      
      case 5: // R1
        left_speed = basic_left_speed + 60;
        right_speed = basic_right_speed - 20;
        break;

      case 6: // R2
        left_speed = basic_left_speed + 85;
        right_speed =  basic_right_speed - 20;
        break;

      case 7: // R3
        left_speed = basic_left_speed + 115;
        right_speed =  basic_right_speed - 20;
        break;

      case 8:
        left_speed = basic_left_speed + 40;
        right_speed = basic_right_speed + 40;
        break;
    
    }

    // if (acc_ON)
    // {
    //   ACCEL();
    //   acc_cnt++;
    //   Serial.print("acc_cnt: ");
    //   Serial.println(acc_cnt);
    // }
    
    if (StopFlag)
    {
      STOP();
      digitalWrite(RED1, HIGH);
      digitalWrite(RED2, HIGH);
    }
    else
    {
      digitalWrite(RED1, LOW);
      digitalWrite(RED2, LOW);
      if (BackFlag)
      {
        GO_BACK();
      }
      else
      {
        GO_FORWARD();
      }
      
    }

    //LED 및 부저 작동함수
    if (BLUE_ON && LED_checkTimer())
    {
      digitalWrite(BLUE1, HIGH);
    }
    else
    {
      BLUE_ON = false;
      digitalWrite(BLUE1, LOW);
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

    // if (show_speed == 10)
    // {
    // Serial.println("-----------------------------");
    // Serial.print("좌측륜 속도: ");
    // Serial.print(left_speed);
    // Serial.print(", ");
    // Serial.print("우측륜 속도: ");
    // Serial.println(right_speed);
    // Serial.println("-----------------------------");
    // show_speed = 0;
    // }
    prev_status = curr_status;
  }
}