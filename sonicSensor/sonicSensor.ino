// Serial communication variables
const int maxChars = 10;
char command[maxChars];
bool haveNewCmd = false;

// Sonic sensor variables
int trigPin = 11; // Trigger
int echoPin = 12; // Echo
long duration, cm, inches;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(5);

  // Set pinmodes
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  delay(100);
  Serial.println("|r|");
}

void loop() {
  if (haveNewCmd == false){
    readSerial();
  }
  else if (haveNewCmd == true){
    execute_command();
    haveNewCmd = false;
  }
  delay(100);
}

void readSerial() {
  char currentChar;
  bool recievingCmd = false;
  int i = 0;

  while (Serial.available() > 0){
    currentChar = Serial.read();
    if (currentChar == byte('<') && recievingCmd == false){
      recievingCmd = true;
    }
    else if (currentChar != byte('>') && recievingCmd == true && i < maxChars){
      command[i] = currentChar;
      i++;
    }
    else {
      command[i] = '\0';
      i = 0;
      recievingCmd = false;
      haveNewCmd = true;
      break;
    }
  }
}

void execute_command() {
  int switchCmd = atoi(command);

  if (switchCmd == 0){
    readSonar();
  }
}

void readSonar() {

  digitalWrite(trigPin, LOW);
  delayMicroseconds(5);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  pinMode(echoPin, INPUT);
  duration= pulseIn(echoPin, HIGH);

  cm = (duration/2) / 29.1;
  Serial.print("|");
  Serial.print(cm);
  Serial.println("|");
}