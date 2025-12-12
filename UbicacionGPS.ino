#include <HardwareSerial.h>
#include <TinyGPSPlus.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <math.h>          // sqrtf, fabsf

// FreeRTOS
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

// =====================================================
// CONFIGURACIÓN WIFI Y SERVIDORES
// =====================================================

const char* WIFI_SSID  = "FAMILIA_MESIAS";
const char* WIFI_PASS  = "cofeeGus314";

const char* WIFI_SSID2 = "POCOGus";      // (por ahora no usado)
const char* WIFI_PASS2 = "12345678.";    // (por ahora no usado)

// Flask: recibe JSON con GPS
const char* GPS_SERVER_URL   = "http://192.168.18.85:5000/gps";

// Servidor que recibe audio
const char* AUDIO_SERVER_URL = "http://192.168.18.85:5000/audio_upload";

// Mutex para que solo haya 1 HTTP POST a la vez
SemaphoreHandle_t httpMutex;

// =====================================================
// GPS (UART + TinyGPS++)
// =====================================================

HardwareSerial GPSSerial(1);
TinyGPSPlus gps;

static const int RXPin   = 16;   // GPS TX -> ESP RX
static const int TXPin   = 17;   // GPS RX -> ESP TX
static const uint32_t GPSBaud = 9600;

// =====================================================
// AUDIO I2S (INMP441) + AGC SUAVE + NOISE GATE
// =====================================================

#define I2S_WS   25   // LRCK / WS
#define I2S_SD   34   // DATA IN
#define I2S_SCK  18   // BCLK / SCK
#define I2S_PORT I2S_NUM_0

#define BUFFER_SIZE 2048
int32_t i2s_buffer[BUFFER_SIZE];         // 32 bits crudos de I2S
int16_t audio_buffer_16[BUFFER_SIZE];    // convertido a int16
uint8_t audio_buffer[BUFFER_SIZE * 2];   // en bytes (PCM16 little-endian)

unsigned long lastSendAudio = 0;
const unsigned long sendIntervalAudio = 50; // ms entre envíos

// ---------- AGC ----------
float currentGain = 1.0f;
const float MAX_GAIN       = 6.0f;   // ganancia máxima
const float MIN_GAIN       = 1.0f;   // ganancia mínima
const float AGC_TARGET_RMS = 0.10f;  // RMS objetivo cuando hay señal fuerte
const float AGC_MIN_RMS    = 0.008f; // por debajo de esto, no tocamos la ganancia
const float AGC_SPEED_UP   = 0.02f;  // rapidez al subir ganancia
const float AGC_SPEED_DOWN = 0.02f;  // rapidez al bajar ganancia

// ---------- Noise gate con histéresis ----------
bool  gateOpen         = false;
float gateGain         = 0.0f;   // 0 (cerrado) → 1 (abierto)
int   loudFrames       = 0;
int   silentFrames     = 0;
const float GATE_OPEN_RMS  = 0.010f;  // si RMS > esto varios frames → abrir
const float GATE_CLOSE_RMS = 0.006f;  // si RMS < esto varios frames → cerrar
const int   FRAMES_TO_OPEN  = 2;      // nº de frames "altos" para abrir
const int   FRAMES_TO_CLOSE = 6;      // nº de frames "silencio" para cerrar
const float GATE_SMOOTH     = 0.9f;   // suavizado de gate (0.9 → suave)

// =====================================================
// PROTOTIPOS
// =====================================================

void audioTask(void* parameter);
void gpsTask(void* parameter);

void setupI2S();
void processAudioWithAGCAndGate(int samples);

void sendGpsToServer(double lat, double lng, int sats, double hdop,
                     const char* fecha, const char* hora, double velocidad);
void SubirCoordenadas();

// =====================================================
// I2S SETUP
// =====================================================

void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num  = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

// =====================================================
// PROCESADO: I2S -> PCM16 + AGC SUAVE + GATE CON HISTÉRESIS
// =====================================================

void processAudioWithAGCAndGate(int samples) {
  float sumSquares = 0.0f;

  // 1) Convertir 32 bits I2S -> int16 y medir RMS
  for (int i = 0; i < samples; i++) {
    // Conversión típica para INMP441 (24 bits en 32)
    int32_t raw = i2s_buffer[i] >> 14;    // igual que en tu calibrador
    int16_t s16 = (int16_t)raw;

    audio_buffer_16[i] = s16;

    float x = (float)s16 / 32768.0f;      // [-1, 1]
    sumSquares += x * x;
  }

  float rms = sqrtf(sumSquares / (samples > 0 ? samples : 1));

  // 2) Actualizar AGC SOLO si hay señal "visible"
  if (rms > AGC_MIN_RMS) {
    if (rms > AGC_TARGET_RMS * 1.2f) {
      // Muy alto → bajar ganancia un poquito
      currentGain *= (1.0f - AGC_SPEED_DOWN);
    } else if (rms < AGC_TARGET_RMS * 0.8f) {
      // Muy bajo (pero por encima de AGC_MIN_RMS) → subir ganancia un poquito
      currentGain *= (1.0f + AGC_SPEED_UP);
    }

    if (currentGain > MAX_GAIN) currentGain = MAX_GAIN;
    if (currentGain < MIN_GAIN) currentGain = MIN_GAIN;
  }

  // 3) Noise gate con histéresis (decidir si puerta abierta/cerrada)
  if (rms > GATE_OPEN_RMS) {
    loudFrames++;
    silentFrames = 0;
  } else if (rms < GATE_CLOSE_RMS) {
    silentFrames++;
    loudFrames = 0;
  } else {
    // zona intermedia → no sumamos a ninguno, pero tampoco reiniciamos del todo
    // (opcional, aquí lo dejamos estable)
  }

  if (!gateOpen && loudFrames >= FRAMES_TO_OPEN) {
    gateOpen = true;
    // Serial.println("[GATE] Abriendo puerta");
  }
  if (gateOpen && silentFrames >= FRAMES_TO_CLOSE) {
    gateOpen = false;
    // Serial.println("[GATE] Cerrando puerta");
  }

  // 4) Suavizado del factor de gate (para evitar "clics")
  float targetGate = gateOpen ? 1.0f : 0.0f;
  gateGain = GATE_SMOOTH * gateGain + (1.0f - GATE_SMOOTH) * targetGate;

  // 5) Aplicar AGC + gate a cada muestra
  for (int i = 0; i < samples; i++) {
    float x = (float)audio_buffer_16[i] / 32768.0f;

    // AGC
    x *= currentGain;

    // Gate suave
    x *= gateGain;

    // Limitar
    if (x > 1.0f)  x = 1.0f;
    if (x < -1.0f) x = -1.0f;

    audio_buffer_16[i] = (int16_t)(x * 32767.0f);
  }

  // 6) Pasar a bytes PCM16 little endian
  for (int i = 0; i < samples; i++) {
    int16_t v = audio_buffer_16[i];
    audio_buffer[i * 2]     =  v        & 0xFF;
    audio_buffer[i * 2 + 1] = (v >> 8) & 0xFF;
  }
}

// =====================================================
// ENVÍO GPS (JSON)
// =====================================================

void sendGpsToServer(double lat, double lng, int sats, double hdop,
                     const char* fecha, const char* hora, double velocidad)
{
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[GPS] WiFi no conectado, no se puede enviar.");
    return;
  }

  if (xSemaphoreTake(httpMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
    Serial.println("[GPS] No pude tomar el mutex HTTP.");
    return;
  }

  HTTPClient http;
  http.begin(GPS_SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  String json = "{";
  json += "\"lat\":";
  json += String(lat, 10);
  json += ",";
  json += "\"lng\":";
  json += String(lng, 10);
  json += ",";
  json += "\"sats\":";
  json += String(sats);
  json += ",";
  json += "\"hdop\":";
  json += String(hdop, 2);
  json += ",";
  json += "\"vel_kmph\":";
  json += String(velocidad, 2);
  json += ",";
  json += "\"hora\":\"";
  json += hora;
  json += "\",";
  json += "\"fecha\":\"";
  json += fecha;
  json += "\"";
  json += "}";

  Serial.print("[GPS] Enviando JSON: ");
  Serial.println(json);

  int httpCode = http.POST(json);
  Serial.print("[GPS] Respuesta HTTP: ");
  Serial.println(httpCode);

  if (httpCode > 0) {
    String payload = http.getString();
    Serial.print("[GPS] Payload: ");
    Serial.println(payload);
  } else {
    Serial.print("[GPS] Error en POST: ");
    Serial.println(http.errorToString(httpCode));
  }

  http.end();
  xSemaphoreGive(httpMutex);
}

void SubirCoordenadas() {
  if (gps.location.isValid()) {
    double lat       = gps.location.lat();
    double lng       = gps.location.lng();
    int    sats      = gps.satellites.value();
    double hdop      = gps.hdop.hdop();
    double velocidad = gps.speed.kmph();

    char fechaBuf[16];
    if (gps.date.isValid()) {
      snprintf(fechaBuf, sizeof(fechaBuf), "%04u-%02u-%02u",
               gps.date.year(), gps.date.month(), gps.date.day());
    } else {
      strcpy(fechaBuf, "0000-00-00");
    }

    char horaBuf[16];
    if (gps.time.isValid()) {
      snprintf(horaBuf, sizeof(horaBuf), "%02u:%02u:%02u",
               gps.time.hour(), gps.time.minute(), gps.time.second());
    } else {
      strcpy(horaBuf, "00:00:00");
    }

    sendGpsToServer(lat, lng, sats, hdop, fechaBuf, horaBuf, velocidad);
  } else {
    Serial.println("[GPS] Posición aún no válida, no se envía.");
  }
}

// =====================================================
// TAREA: AUDIO (CORE 0)
// =====================================================

void audioTask(void* parameter) {
  Serial.println("[AUDIO] Tarea de audio iniciada en núcleo " + String(xPortGetCoreID()));

  unsigned long lastStats = 0;

  for (;;) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[AUDIO] WiFi desconectado. Reintentando...");
      WiFi.reconnect();
      vTaskDelay(pdMS_TO_TICKS(5000));
      continue;
    }

    unsigned long now = millis();
    if (now - lastSendAudio < sendIntervalAudio) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }
    lastSendAudio = now;

    size_t bytes_read = 0;
    esp_err_t result = i2s_read(
      I2S_PORT,
      &i2s_buffer,
      sizeof(i2s_buffer),
      &bytes_read,
      portMAX_DELAY
    );

    if (result != ESP_OK || bytes_read == 0) {
      Serial.println("[AUDIO] Error al leer I2S");
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    int samples = bytes_read / 4;   // 4 bytes por muestra (32 bits)
    processAudioWithAGCAndGate(samples);

    // Enviar audio
    if (xSemaphoreTake(httpMutex, pdMS_TO_TICKS(2000)) == pdTRUE) {
      HTTPClient http;
      http.begin(AUDIO_SERVER_URL);
      http.addHeader("Content-Type", "application/octet-stream");
      http.addHeader("X-Audio-Format", "pcm16");
      http.addHeader("X-Sample-Rate", "16000");

      int httpCode = http.POST(audio_buffer, samples * 2);

      if (httpCode > 0) {
        if (now - lastStats >= 1000) {
          Serial.print("[AUDIO] Enviado: ");
          Serial.print(samples * 2);
          Serial.print(" bytes | HTTP: ");
          Serial.print(httpCode);
          Serial.print(" | Ganancia: ");
          Serial.print(currentGain, 2);
          Serial.print("x | Gate: ");
          Serial.print(gateGain, 2);
          Serial.print(" | WiFi: ");
          Serial.print(WiFi.RSSI());
          Serial.println(" dBm");
          lastStats = now;
        }
      } else {
        Serial.print("[AUDIO] Error HTTP: ");
        Serial.println(http.errorToString(httpCode));
      }

      http.end();
      xSemaphoreGive(httpMutex);
    }

    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// =====================================================
// TAREA: GPS (CORE 1)
// =====================================================

void gpsTask(void* parameter) {
  Serial.println("[GPS] Tarea de GPS iniciada en núcleo " + String(xPortGetCoreID()));

  unsigned long lastSendGps = 0;

  for (;;) {
    // Leer NMEA
    while (GPSSerial.available() > 0) {
      char c = GPSSerial.read();
      gps.encode(c);
    }

    if (gps.location.isUpdated()) {
      Serial.println("===== NUEVA POSICION =====");
      Serial.print("Latitud : ");
      Serial.println(gps.location.lat(), 10);
      Serial.print("Longitud: ");
      Serial.println(gps.location.lng(), 10);
      Serial.print("HDOP    : ");
      Serial.println(gps.hdop.hdop());
      Serial.print("Sats    : ");
      Serial.println(gps.satellites.value());
      Serial.println();
    }

    if (gps.date.isUpdated() && gps.time.isUpdated()) {
      Serial.print("Fecha: ");
      Serial.print(gps.date.day());
      Serial.print('/');
      Serial.print(gps.date.month());
      Serial.print('/');
      Serial.print(gps.date.year());
      Serial.print("  Hora (UTC): ");

      if (gps.time.hour() < 10)  Serial.print('0');
      Serial.print(gps.time.hour());
      Serial.print(':');

      if (gps.time.minute() < 10)  Serial.print('0');
      Serial.print(gps.time.minute());
      Serial.print(':');

      if (gps.time.second() < 10)  Serial.print('0');
      Serial.println(gps.time.second());
    }

    // Enviar al servidor cada 5 s
    unsigned long now = millis();
    if (now - lastSendGps >= 5000) {
      SubirCoordenadas();
      lastSendGps = now;
    }

    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

// =====================================================
// SETUP & LOOP
// =====================================================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n=================================");
  Serial.println("ESP32: Audio I2S + GPS (doble núcleo)");
  Serial.println("Modo: AGC suave + noise gate con histéresis");
  Serial.println("=================================\n");

  // UART para GPS
  GPSSerial.begin(GPSBaud, SERIAL_8N1, RXPin, TXPin);

  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando a WiFi");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi conectado!");
    Serial.print("IP ESP32: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n✗ No se pudo conectar al WiFi, reiniciando en 5s...");
    delay(5000);
    ESP.restart();
  }

  // I2S
  Serial.println("✓ Configurando I2S...");
  setupI2S();
  Serial.println("✓ I2S listo\n");

  // Mutex HTTP
  httpMutex = xSemaphoreCreateMutex();

  // Crear tareas
  xTaskCreatePinnedToCore(
    audioTask,
    "AudioTask",
    8192,
    NULL,
    1,
    NULL,
    0   // núcleo 0
  );

  xTaskCreatePinnedToCore(
    gpsTask,
    "GpsTask",
    8192,
    NULL,
    1,
    NULL,
    1   // núcleo 1
  );
}

void loop() {
  // No hacemos nada aquí, todo corre en las tareas
  vTaskDelay(pdMS_TO_TICKS(1000));
}

