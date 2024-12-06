// person 수신 부저 작동
// red 수신 LED 점등
// run 수신 전진 _ rev(L) = rev(R) = 150
// stop 수신 정지 _ rev(L) = rev(R) = 0
// reverse 수신 후진 _ rev(L) = rev(R) = 150
// l1, l2, l3 수신 좌회전 _ rev(L) > rev(R)
// r1, r2, r3 수신 우회전 _ rev(L) < rev(R)


#include <WiFi.h>
// #include <String>


// const char* ssid = "Addinedu_509_office_5G";  // Wi-Fi SSID
// const char* password = "Addinedu8565!";        // Wi-Fi 비밀번호

const char* ssid = "Senna";  // 핫스팟
const char* password = "159753ff";   // 핫스팟 비밀번호

const int LED = 13;
const int BUZZER = 4;

const int speed = 100;
const int basic_right_speed = speed;
const int basic_left_speed = basic_right_speed + 40;

int right_speed;
int left_speed;

// int prev_right_speed;
// int prev_left_speed;

// LED, 부저 millis() 세팅
unsigned long LED_start_time = 0;
const unsigned long LED_DELAY_TIME = 5000;

unsigned long BUZZER_start_time = 0;
const unsigned long BUZZER_DELAY_TIME = 1000;

bool RED_ON = false;
bool BUZZER_ON = false;

const int motor_R_forward = 26;  // 오른쪽 정방향 IN1
const int motor_R_reverse = 27;  // 오른쪽 역박향 IN2

const int motor_L_forward = 33;  // 왼쪽 정방향 IN3
const int motor_L_reverse = 25;  // 왼쪽 역박향 IN4

// bool Left1 = false;
// bool Left2 = false;
// bool Left3 = false;
// bool Right1 = false;
// bool Right2 = false;
// bool Right3 = false;

// 서버로부터 정보 전달받는 변수
// const int MAX_PARTS = 4;
// String data[MAX_PARTS];
// int splitCount = 0;


// 속도 출력용 변수
int show_speed = 0;

WiFiServer server(8080);


// void splitString(String data, String delimiter, String result[], int& count)
// {
//   int startIndex = 0;
//   int endIndex = 0;
//   count = 0;

//   while ((endIndex = data.indexOf(delimiter, startIndex)) != -1) {
//     result[count++] = data.substring(startIndex, endIndex);
//     startIndex = endIndex + delimiter.length();

//     if (count >= MAX_PARTS) {
//       break; // 배열 크기 초과 방지
//     }
//   }

//   result[count++] = data.substring(startIndex);
// }

void LED_startTimer()
  {
    LED_start_time = millis();
  }
  void BUZZER_startTimer()
  {
    BUZZER_start_time = millis();
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

  bool BUZZER_checkTimer()
  {
    unsigned long now = millis();
    if (BUZZER_start_time == 0)
    {
      return false;
    }
    else if((now - BUZZER_start_time) <= BUZZER_DELAY_TIME)
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

  void Go_BACK()
  {
      analogWrite(motor_R_forward, 0);
      analogWrite(motor_R_reverse, right_speed);

      analogWrite(motor_L_forward, 0);
      analogWrite(motor_L_reverse, left_speed);
  }

  void STOP()
  {
      left_speed = 0;
      right_speed = 0;
      analogWrite(motor_R_forward, right_speed);
      analogWrite(motor_R_reverse, right_speed);

      analogWrite(motor_L_forward, left_speed);
      analogWrite(motor_L_reverse, left_speed);
  }

  void R1(int prev_right_speed, int prev_left_speed)
  {
    left_speed = prev_left_speed + 40;
    right_speed = prev_right_speed;

    GO_FORWARD();
  }
    void R2(int prev_right_speed, int prev_left_speed)
  {
    left_speed = prev_left_speed + 60;
    right_speed = prev_right_speed;
    GO_FORWARD();
  }
    void R3(int prev_right_speed, int prev_left_speed)
  {
    left_speed = prev_left_speed + 80;
    right_speed = prev_right_speed;
    GO_FORWARD();
  }
      
  

  void L1(int prev_right_speed, int prev_left_speed)
  {
    right_speed = prev_right_speed + 80;
    left_speed = prev_left_speed - 20;

    GO_FORWARD();
  }
  void L2(int prev_right_speed, int prev_left_speed)
  {
    right_speed = prev_right_speed + 100;
    left_speed = prev_left_speed - 20;

    GO_FORWARD();
  }
  void L3(int prev_right_speed, int prev_left_speed)
  {
    right_speed = prev_right_speed + 120;
    left_speed = prev_left_speed - 20;

    GO_FORWARD();
  }

void setup() {
  Serial.begin(115200);  // 시리얼 모니터 시작
  
  pinMode(LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);

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
      int prev_right_speed;
      int prev_left_speed;

      case 0: // 연결 확인
        client.println("connected");
        break;
      
      case 1: // 차주 인식
        client.println("welcome soyoung");
        RED_ON = true;
        LED_startTimer();
        break;

      case 2: // 전진
        client.println("drive");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;

        GO_FORWARD();
        break;

      case 3: // 정지
        client.println("stop");
        BUZZER_ON = true;
        BUZZER_startTimer();
        STOP();
        break;

      case 4: // 후진
        client.println("reverse");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;
        Go_BACK();
        break;
      
      case 5: // L1
        client.println("L1");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;

        prev_right_speed = right_speed;
        prev_left_speed = left_speed;
        L1(prev_right_speed, prev_left_speed);
        
        break;
      
      case 6: // L2
        client.println("L2");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;

        prev_right_speed = right_speed;
        prev_left_speed = left_speed;

        L2(prev_right_speed, prev_left_speed);
        break;

      case 7: // L3
        client.println("L3");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;

        prev_right_speed = right_speed;
        prev_left_speed = left_speed;
        L3(prev_right_speed, prev_left_speed);
        break;

      case 8: // R1
        client.println("R1");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;
        
        prev_left_speed = left_speed;
        prev_right_speed = right_speed;

        R1(prev_right_speed, prev_left_speed);
        break;

      case 9: // R2
        client.println("R2");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;

        prev_left_speed = left_speed;
        prev_right_speed = right_speed;
        R2(prev_right_speed, prev_left_speed);
        break;
      
      case 10: // R3
        client.println("R3");
        left_speed = basic_left_speed;
        right_speed = basic_right_speed;

        prev_left_speed = left_speed;
        prev_right_speed = right_speed;

        R3(prev_right_speed, prev_left_speed);
        break;

    }

    //LED 및 부저 작동함수
    if (RED_ON && LED_checkTimer())
    {
      digitalWrite(LED, HIGH);
    }
    else
    {
      digitalWrite(LED, LOW);
    }

    if (BUZZER_ON && BUZZER_checkTimer())
    {
      tone(BUZZER, 500);
    }
    else
    {
      tone(BUZZER, 0);
    }


    if (show_speed == 10)
    {
    Serial.println("-----------------------------");
    Serial.print("좌측륜 속도: ");
    Serial.print(left_speed);
    Serial.print(", ");
    Serial.print("우측륜 속도: ");
    Serial.println(right_speed);
    Serial.println("-----------------------------");
    show_speed = 0;
    }
    

  }
}
