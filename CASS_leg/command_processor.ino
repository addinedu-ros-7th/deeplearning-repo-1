#include <WiFi.h>

// const char* ssid = "Addinedu_509_office_5G";  // Wi-Fi SSID777
// const char* password = "Addinedu8565!";        // Wi-Fi 비밀번호

const char* ssid = "Senna";  // 핫스팟
const char* password = "159753ff";   // 핫스팟 비밀번호

const int RED1 = 13;
const int RED2 = 4;
const int BLUE1 = 18;
 
const int speed = 90;
int steering_mode;

const int basic_right_speed = speed;
const int basic_left_speed = basic_right_speed + 40;

int right_speed;
int left_speed;
int acc_cnt; // 우측 바퀴 가속 보정용 count 변수

// LED, 부저 millis() 세팅
unsigned long LED_start_time = 0;
const unsigned long LED_DELAY_TIME = 5000;

// unsigned long BUZZER_start_time = 0;
// const unsigned long BUZZER_DELAY_TIME = 1000;

bool BLUE_ON = false;

const int motor_R_forward = 26;  // 오른쪽 정방향 IN1
const int motor_R_reverse = 27;  // 오른쪽 역박향 IN2

const int motor_L_forward = 21;  // 왼쪽 정방향 IN3
const int motor_L_reverse = 19;  // 왼쪽 역박향 IN4

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
  // void BUZZER_startTimer()
  // {
  //   BUZZER_start_time = millis();
  // }

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

      case 31: // 전진
        client.println("drive");
        BackFlag = false;
        StopFlag = false;
        steering_mode = 1;
        break;

      case 32: // 정지
        client.println("stop");
        steering_mode = 0;
        BackFlag = false;
        StopFlag = true;
        //BUZZER_ON = true;
        // BUZZER_startTimer();
        break;

      case 33: // 후진
        client.println("reverse");
        steering_mode = 1;
        StopFlag = false;
        BackFlag = true;
        break;
      
      case 41: // L1
        client.println("L1");
        if (StopFlag == false)
        {
          steering_mode = 2;
        }
        else
        {
          client.println("Breaking Now. L1 unavailable");
        }     
        break;
      
      case 42: // L2
        client.println("L2");
        if (StopFlag == false)
        {
          steering_mode = 3;
        }
        else
        {
          client.println("Breaking Now. L2 unavailable");
        }
        break;

      case 43: // L3
        client.println("L3");
        if (StopFlag == false)
        {
          steering_mode = 4;
        }
        else
        {
          client.println("Breaking Now. L3 unavailable");
        }
        break;

      case 51: // R1
        client.println("R1");
        if (StopFlag == false)
        {
          steering_mode = 5;
        }
        else
        {
          client.println("Breaking Now. R1 unavailable");
        }
        break;

      case 52: // R2
        client.println("R2");
        if (StopFlag == false)
        {
          steering_mode = 6;
        }
        else
        {
          client.println("Breaking Now. R2 unavailable");
        }
        break;
      
      case 53: // R3
        client.println("R3");
        if (StopFlag == false)
        {
          steering_mode = 7;
        }
        else
        {
          client.println("Breaking Now. R3 unavailable");
        }
        break;

      case 99: // 비상 제동
        client.println("Emergency Breaking");
        Emergency_Signal = true;
    }

    switch (steering_mode) {

      case 0: // 정지
        left_speed = 0;
        right_speed = 0;

      case 1: // 기본 속도
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;
        break;

      case 2: // L1
        right_speed = basic_right_speed + 60;
        left_speed = basic_left_speed - 30;
        break;

      case 3: // L2
        right_speed = basic_right_speed + 80;
        left_speed = basic_left_speed - 30;
        break;

      case 4: // L3
        right_speed = basic_right_speed + 100;
        left_speed = basic_left_speed - 30;
        break;
      
      case 5: // R1
        left_speed = basic_left_speed + 30;
        right_speed = basic_right_speed;
        break;

      case 6: // R2
        left_speed = basic_left_speed + 60;
        right_speed = basic_right_speed;
        break;

      case 7: // R3
        left_speed = basic_left_speed + 110;
        right_speed = basic_right_speed;
        break;
    }

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
    // Serial.print("RED_ON: ");
    // Serial.println(RED_ON);
    // Serial.print("LED_checkTimer: ");
    // Serial.println(LED_checkTimer());


  }
}
