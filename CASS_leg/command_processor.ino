// person 수신 부저 작동
// red 수신 LED 점등
// run 수신 전진 _ rev(L) = rev(R) = 150
// stop 수신 정지 _ rev(L) = rev(R) = 0
// reverse 수신 후진 _ rev(L) = rev(R) = 150
// left 수신 좌회전 _ rev(L) > rev(R)
// right 수신 우회전 _ rev(L) < rev(R)


#include <WiFi.h>
// #include <String>


// const char* ssid = "Addinedu_509_office_5G";  // Wi-Fi SSID
// const char* password = "Addinedu8565!";        // Wi-Fi 비밀번호

const char* ssid = "Senna";  // 핫스팟
const char* password = "159753ff";   // 핫스팟 비밀번호

const int LED = 13;
const int BUZZER = 4;
const int speed = 120;
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

bool turn_right = false;
bool turn_left = false;

// 서버로부터 정보 전달받는 변수
// const int MAX_PARTS = 4;
// String data[MAX_PARTS];
// int splitCount = 0;

// 차륜 회전속도 조절용 변수
int speed_change_count = 0;
int left_speed;
int prev_left_speed;
int right_speed;
int prev_right_speed;

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

  void TURN_RIGHT(int prev_left_speed)
  {
    left_speed = prev_left_speed + 60;
    // 가속 후 복귀
    // if (speed_change_count >= 0 && speed_change_count < 20)
    // {
    //   left_speed += 4;
    //   // right_speed -= 2;
    //   speed_change_count ++;
    // }
    // else if (speed_change_count >= 20 && speed_change_count < 40)
    // {
    //   left_speed -= 4;
    //   // right_speed += 2;
    //   speed_change_count ++;
    // }
    // if (speed_change_count >= 40)
    // {
    //   speed_change_count = 0;
    //   turn_right = false;
    // }
//==========================================================
//  가속 후 복귀 X 
    // if (speed_change_count >= 0 && speed_change_count <=15) 
    // {
    //   right_speed += 2;
    //   left_speed -= 2;
    //   speed_change_count ++;
    // }
    // else
    // {
    //   speed_change_count = 0;
    //   turn_right = false;
    // }

    GO_FORWARD();
  }
      
  

  void TURN_LEFT(int prev_right_speed)
  {
    right_speed = prev_right_speed + 60;
    // 가속 후 복귀
    // if (speed_change_count >= 0 && speed_change_count < 20)
    // {
    //   // left_speed -= 2;
    //   right_speed += 4;
    //   speed_change_count ++;
    // }
    // else if (speed_change_count >= 20 && speed_change_count < 40)
    // {
    //   // left_speed += 2;
    //   right_speed -= 4;
    //   speed_change_count ++;
    // }
    // if (speed_change_count >= 40)
    // {
    //   speed_change_count = 0;
    //   turn_left = false;
    // }
//======================================================================
// 가속 후 복귀X
    // if (speed_change_count >= 0 && speed_change_count <=15) 
    // {
    //   right_speed -= 2;
    //   left_speed += 2;
    //   speed_change_count ++;
    // }
    // else
    // {
    //   speed_change_count = 0;
    //   turn_left = false;
    // }

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

  right_speed = speed;
  left_speed = right_speed+35;
}

void loop() {
  WiFiClient client = server.available();

  while (client.connected())
  {
    show_speed ++;
    client.setTimeout(20);
    String data = client.readStringUntil('\n'); // 줄바꿈("\n") 기준으로 데이터 읽기 
    data.trim(); // 데이터 앞뒤 공백 제거
    // if (receivedData != 0)
    // {
    //   splitString(receivedData, "&&", data, splitCount);
    // }
    // else
    // {
    //   Serial.println(receivedData);
    // }

    if (data != 0)
    {
      Serial.println(data);
    }
    delay(50);

    if (data == "connect")
    {
      client.println("connected");
    }

    // LED 및 부저 작동
    if (data == "soyoung")
    {
      RED_ON = true;
      LED_startTimer();
      client.println("welcome soyoung");
    }
    if (data == "person")
    {
      BUZZER_startTimer();
      BUZZER_ON = true;
      client.println("person detected");
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

    // 모터 작동

    // 전진
    if (data == "d")
    {
      client.println("drive");
      left_speed = 0;
      right_speed = 0;
      GO_FORWARD();
      Serial.print("drive");
      right_speed = speed;
      left_speed = speed + 35;
      GO_FORWARD();
    }

    // 후진
    else if (data == "r")
    {
      client.println("reverse");
      left_speed = 0;
      right_speed = 0;
      Go_BACK();

      Serial.print("reverse");
      right_speed = speed;
      left_speed = right_speed+35;
      
      Go_BACK();
    }
    
    // 정지
    else if (data== "stop")
    {
      client.println("stop");
      turn_right = false;
      turn_left = false;
      Serial.print("stop");
      STOP();
    }

    // 좌회전
    else if (data == "tl")
    {
      if (turn_right != true)
      {
        prev_right_speed = right_speed;
        turn_left = true;
      }
      else
      {
        turn_left = false;
      }
    }

    // 우회전
    else if (data == "tr")
    {
      if (turn_left != true)
      {
        prev_left_speed = left_speed;
        turn_right = true;
      }
      else
      {
        turn_right = false;
      }
    }


    else if (data == "up")
    {
      left_speed += 10;
      right_speed += 10;
      GO_FORWARD();
    }

    else if (data == "down")
    {
      left_speed -= 10;
      right_speed -= 10;
      GO_FORWARD();
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
    
    if (turn_right == true)
    {
      TURN_RIGHT(prev_left_speed);
    }

    if (turn_left == true)
    {
      TURN_LEFT(prev_right_speed);
    }

    // Serial.println(speed_change_count);
    // Serial.print("turn_left: ");
    // Serial.println(turn_left);
    // Serial.print("turn_right: ");
    // Serial.println(turn_right);
  }
}
