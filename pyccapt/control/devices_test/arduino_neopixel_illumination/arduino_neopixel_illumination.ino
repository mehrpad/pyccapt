/*
 * PyCCAPT camera-alignment illumination controller.
 *
 * Hardware:
 *   Arduino Nano D6 -- 330..470 ohm resistor -- DIN of the first ring
 *   24-pixel ring DOUT -- DIN of 12-pixel ring -- DIN of second 12-pixel ring
 *   All rings powered from the external 5 V supply in parallel.
 *   Arduino GND and external-supply GND must be connected together.
 *
 * Install the "Adafruit NeoPixel" library through Arduino IDE ->
 * Tools -> Manage Libraries before uploading this sketch.
 *
 * Serial protocol at 115200 baud:
 *   PING              -> PYCCAPT_ILLUMINATION
 *   ON <0..100>       -> turn white illumination on at this percentage
 *   OFF               -> turn illumination off
 *   BRIGHTNESS <0..100> -> update brightness; applies immediately if on
 */

#include <Adafruit_NeoPixel.h>

constexpr uint8_t DATA_PIN = 6;
constexpr uint16_t PIXEL_COUNT = 48;

Adafruit_NeoPixel pixels(PIXEL_COUNT, DATA_PIN, NEO_GRB + NEO_KHZ800);

uint8_t brightnessPercent = 50;
bool illuminationOn = false;
String commandBuffer;

void applyIllumination() {
  pixels.setBrightness(map(brightnessPercent, 0, 100, 0, 255));
  if (illuminationOn) {
    pixels.fill(pixels.Color(255, 255, 255));
  } else {
    pixels.clear();
  }
  pixels.show();
}

int clampPercent(int value) {
  return constrain(value, 0, 100);
}

void processCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "PING") {
    Serial.println("PYCCAPT_ILLUMINATION");
    return;
  }
  if (command == "OFF") {
    illuminationOn = false;
    applyIllumination();
    Serial.println("OK");
    return;
  }
  if (command.startsWith("ON ")) {
    brightnessPercent = clampPercent(command.substring(3).toInt());
    illuminationOn = true;
    applyIllumination();
    Serial.println("OK");
    return;
  }
  if (command.startsWith("BRIGHTNESS ")) {
    brightnessPercent = clampPercent(command.substring(11).toInt());
    applyIllumination();
    Serial.println("OK");
    return;
  }
  Serial.println("ERROR");
}

void setup() {
  Serial.begin(115200);
  pixels.begin();
  illuminationOn = true;  // Safe default: illumination is available at startup.
  applyIllumination();    // Default brightness is 50%.
}

void loop() {
  while (Serial.available()) {
    const char received = static_cast<char>(Serial.read());
    if (received == '\n' || received == '\r') {
      if (commandBuffer.length() > 0) {
        processCommand(commandBuffer);
        commandBuffer = "";
      }
    } else {
      commandBuffer += received;
    }
  }
}
