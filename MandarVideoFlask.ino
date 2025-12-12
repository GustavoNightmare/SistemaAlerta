#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// =====================
// WIFI + FLASK
// =====================
const char* WIFI_SSID = "FAMILIA_MESIAS";
const char* WIFI_PASS = "cofeeGus314";

// Ejemplo: "http://192.168.1.50:5000/video"
const char* FLASK_URL = "http://192.168.18.85:5000/video";

// Cada cuánto enviar un frame (ms). 100ms ~ 10 FPS, 200ms ~ 5 FPS
static const uint32_t FRAME_INTERVAL_MS = 150;

// Límite de tu Flask (MAX_FRAME_SIZE = 500000)
static const size_t MAX_FRAME_BYTES = 500000;

// =====================
// Pines cámara (Freenove ESP32-S3 WROOM)
// =====================
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1

#define XCLK_GPIO_NUM   15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5

#define Y9_GPIO_NUM     16
#define Y8_GPIO_NUM     17
#define Y7_GPIO_NUM     18
#define Y6_GPIO_NUM     12
#define Y5_GPIO_NUM     10
#define Y4_GPIO_NUM     8
#define Y3_GPIO_NUM     9
#define Y2_GPIO_NUM     11

#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM   13

#define LED_ON_GPIO     2

// =====================
// Tuning sensor
// =====================
static void applyOv5640Tuning() {
  sensor_t *s = esp_camera_sensor_get();
  s->set_exposure_ctrl(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_awb_gain(s, 1);
  s->set_whitebal(s, 1);

  s->set_brightness(s, 1);   // -2..2
  s->set_contrast(s, 1);     // -2..2
  s->set_saturation(s, 0);   // -2..2

  // Orientación (como ya lo tenías)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
}

// =====================
// Enviar frame a Flask
// =====================
static bool postFrameToFlask(const uint8_t* buf, size_t len) {
  if (WiFi.status() != WL_CONNECTED) return false;

  WiFiClient client;
  HTTPClient http;

  http.setTimeout(4000); // ms
  if (!http.begin(client, FLASK_URL)) {
    Serial.println("[HTTP] begin() fallo");
    return false;
  }

  http.addHeader("Content-Type", "image/jpeg");
  // Opcional:
  // http.addHeader("Connection", "close");

  int code = http.POST((uint8_t*)buf, len);
  if (code > 0) {
    // Flask responde 'OK' con 200
    if (code != 200) {
      Serial.printf("[HTTP] Resp code: %d\n", code);
      String payload = http.getString();
      if (payload.length()) {
        Serial.print("[HTTP] Body: ");
        Serial.println(payload);
      }
      http.end();
      return false;
    }
  } else {
    Serial.printf("[HTTP] POST error: %s\n", http.errorToString(code).c_str());
    http.end();
    return false;
  }

  http.end();
  return true;
}

// =====================
// Setup
// =====================
void setup() {
  Serial.begin(115200);
  delay(800);

  pinMode(LED_ON_GPIO, OUTPUT);
  digitalWrite(LED_ON_GPIO, HIGH);

  Serial.println("\n=== ESP32-S3 OV5640 -> Flask /video ===");
  Serial.printf("psramFound(): %s\n", psramFound() ? "SI" : "NO");
  Serial.printf("PSRAM size: %u bytes\n", ESP.getPsramSize());

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  WiFi.setSleep(false);

  Serial.print("Conectando WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("IP ESP32: ");
  Serial.println(WiFi.localIP());

  // Config cámara
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // OJO con tamaño vs MAX_FRAME_SIZE del server
  config.frame_size   = FRAMESIZE_VGA;   // prueba QVGA si se pasa de peso
  config.jpeg_quality = 12;              // sube a 14-18 si te rechaza por size
  config.grab_mode    = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count    = 2;
  } else {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count    = 1;
    config.frame_size  = FRAMESIZE_QVGA;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("ERROR: esp_camera_init fallo: 0x%x\n", err);
    while (true) delay(1000);
  }

  sensor_t *s = esp_camera_sensor_get();
  Serial.printf("Sensor PID: 0x%X\n", s->id.PID);

  applyOv5640Tuning();

  Serial.print("Enviando a Flask: ");
  Serial.println(FLASK_URL);
}

// =====================
// Loop
// =====================
void loop() {
  static uint32_t lastSend = 0;
  const uint32_t now = millis();

  if (now - lastSend < FRAME_INTERVAL_MS) return;
  lastSend = now;

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ fb NULL");
    delay(10);
    return;
  }

  // Si por alguna razón se pasa del límite del server, lo saltamos
  if (fb->len > MAX_FRAME_BYTES) {
    Serial.printf("⚠️ Frame muy grande (%u bytes). Baja frame_size o sube jpeg_quality.\n", (unsigned)fb->len);
    esp_camera_fb_return(fb);
    return;
  }

  bool ok = postFrameToFlask(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  if (!ok) {
    Serial.println("⚠️ No se pudo enviar frame a Flask");
  } else {
    Serial.printf("✅ Frame enviado (%u bytes)\n", (unsigned)fb->len);
  }
}
