#include <Arduino.h>
#include <MicroTFLite.h>

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3
#endif

#include "glucose.h"  // glucose_tflite[] and glucose_tflite_len
struct Stats;         // forward decl

#include <FS.h>
#include <SPIFFS.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <time.h>
#include "mbedtls/sha1.h"


#define USE_OLED 0

// ===================== CONFIG =====================
static constexpr int    kUartBaud        = 115200;
static constexpr size_t kTensorArenaSize = 96 * 1024;  // PSRAM build
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Model window/scale
static constexpr int    WIN   = 12;
static constexpr float  MIN_G = 40.0f;
static constexpr float  MAX_G = 300.0f;

// setting to 1 to convert to mg/dL
#define OUTPUT_RANGE_ZERO_TO_ONE 1

// If 1: feed RAW mg/dL to the model input tensor
// If 0: feed normalized 0..1 to the model input tensor
#define INPUT_IS_RAW_MGDL 0


// Analog sensor pin (potentiometer wiper)
#define SENSOR_PIN 1   // GPIO1 on ESP32-S3 DevKitC-1

// OLED (I2C on default S3 pins: SDA=8, SCL=9), address 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
static Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ===================== NIGHTSCOUT CONFIG =====================
const char* WIFI_SSID = "<Wifi ssid for ESP32 to connect to>";
const char* WIFI_PASS = "<Wifi password>";

// Nightscout host
const char* NIGHTSCOUT_HOST = "<Nightscout public network url>";

// Nightscout API_SECRET (plaintext).
const char* API_SECRET_PLAINTEXT = "<plain text API_SECRET of Nightscout>";


// ===================== GLOBALS =====================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
using ResolverT = tflite::MicroMutableOpResolver<24>;
static ResolverT resolver;

TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

// For Serial Plotter
static bool  g_plot_header_printed = false;
static float g_last_input_mgdl     = NAN;

// Rolling buffer (last WIN mg/dL readings)
static float g_ring[WIN];
static int   g_count = 0;      // number of valid samples (<= WIN)
static int   g_head  = 0;      // next write index

// ---- Logging (SPIFFS) ----
static bool g_log_enabled = false;
static const char* kLogPath = "/glucose_log.csv";

// ---- Analog sampling state ----
static bool     g_sampling   = false;   // automatic sampling ON/OFF
static uint32_t g_period_ms  = 1000;    // default 1s
static uint32_t g_next_ms    = 0;

// Simple linear calibration (mv to mg/dL)
static float g_lo_mv   = 0.0f,    g_lo_mgdl = 40.0f;
static float g_hi_mv   = 3300.0f, g_hi_mgdl = 300.0f;

// ===================== UTILS =====================
static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

static float g_last_pred_mgdl = NAN;


// ===================== ACCURACY HELPERS =====================
#include <math.h>  // fabsf, sqrtf

struct Stats { float mae; float rmse; int n; };

static inline void resetStats(Stats &s) {
  s.mae = 0.0f; s.rmse = 0.0f; s.n = 0;
}

static inline void addSample(Stats &s, float y_true, float y_pred) {
  float e = y_pred - y_true;
  s.mae  += fabsf(e);
  s.rmse += e * e;
  s.n    += 1;
}

static inline void finishStats(Stats &s) {
  if (s.n > 0) { s.mae /= s.n; s.rmse = sqrtf(s.rmse / s.n); }
}

// ===================== LOGGING =====================
void logLine(float in_mgdl, float pred_mgdl) {
  if (!g_log_enabled) return;
  static bool header_written = false;
  File f = SPIFFS.open(kLogPath, FILE_APPEND);
  if (!f) { Serial.println(F("[log] open failed")); return; }
  if (!header_written && f.size() == 0) {
    f.println("ms,in_mgdl,pred_mgdl");
    header_written = true;
  }
  f.printf("%lu,%.2f,%.2f\n", (unsigned long)millis(), in_mgdl, pred_mgdl);
  f.close();
}

// ===================== TFLM/PRINT HELPERS =====================
void haltWithMessage(const __FlashStringHelper* msg) {
  Serial.println(msg);
  while (true) { delay(1000); }
}

void printTensorShape(const char* name, const TfLiteTensor* t) {
  Serial.print(name); Serial.print(" shape: [");
  if (t && t->dims) {
    for (int i = 0; i < t->dims->size; i++) {
      Serial.print(t->dims->data[i]);
      if (i < t->dims->size - 1) Serial.print(", ");
    }
  }
  Serial.println("]");
}

void plotLine(float in_mgdl, float pred_mgdl) {
  if (!g_plot_header_printed) {
    Serial.println(F("in_mgdl\tpred_mgdl"));
    g_plot_header_printed = true;
  }
  Serial.print(in_mgdl, 2);
  Serial.print('\t');
  Serial.println(pred_mgdl, 2);
}

float dequantizeFirst(const TfLiteTensor* out) {
  if (!out) return NAN;
  if (out->type == kTfLiteFloat32) {
    return reinterpret_cast<float*>(out->data.raw)[0];
  } else if (out->type == kTfLiteInt8) {
    const int8_t q = reinterpret_cast<int8_t*>(out->data.raw)[0];
    return (q - out->params.zero_point) * out->params.scale;
  }
  return NAN;
}

// Print outputs and update Plotter + OLED + log
void printOutputSamples(const TfLiteTensor* out) {
  if (!out) { Serial.println(F("No output tensor.")); return; }

  int elem_size = 0;
  switch (out->type) {
    case kTfLiteFloat32: elem_size = sizeof(float); break;
    case kTfLiteInt8:    elem_size = sizeof(int8_t); break;
    default:
      Serial.print(F("Unsupported output type: ")); Serial.println(out->type);
      return;
  }
  int n_elems  = out->bytes / elem_size;
  int out_count = (n_elems < 4) ? n_elems : 4;

  Serial.println(F("Invoke OK. Output sample(s):"));
  for (int i = 0; i < out_count; i++) {
    if (out->type == kTfLiteFloat32) {
      float y = reinterpret_cast<float*>(out->data.raw)[i];
      Serial.print(F("  y[")); Serial.print(i); Serial.print(F("] = "));
      Serial.println(y, 6);
#if OUTPUT_RANGE_ZERO_TO_ONE
      float mgdl = y * (MAX_G - MIN_G) + MIN_G;
      Serial.print(F("      ≈ ")); Serial.print(mgdl, 2); Serial.println(F(" mg/dL"));
#endif
    } else if (out->type == kTfLiteInt8) {
      int8_t q = reinterpret_cast<int8_t*>(out->data.raw)[i];
      float y  = (q - out->params.zero_point) * out->params.scale;
      Serial.print(F("  y_q[")); Serial.print(i); Serial.print(F("] = "));
      Serial.print(q);
      Serial.print(F("  → y[")); Serial.print(i); Serial.print(F("] = "));
      Serial.println(y, 6);
#if OUTPUT_RANGE_ZERO_TO_ONE
      float mgdl = y * (MAX_G - MIN_G) + MIN_G;
      Serial.print(F("      ≈ ")); Serial.print(mgdl, 2); Serial.println(F(" mg/dL"));
#endif
    }
  }

  // First element drives Plotter/OLED/log
  float y0_norm = dequantizeFirst(out);
  float pred_mgdl = y0_norm;
#if OUTPUT_RANGE_ZERO_TO_ONE
  pred_mgdl = y0_norm * (MAX_G - MIN_G) + MIN_G;
#endif
  g_last_pred_mgdl = pred_mgdl;
  float in_mgdl = isnan(g_last_input_mgdl) ? 0.0f : g_last_input_mgdl;

  plotLine(in_mgdl, pred_mgdl);
  logLine(in_mgdl, pred_mgdl);

  // OLED live update
  #if USE_OLED
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.print(F("Pred: "));
    display.println((int)pred_mgdl);
    display.setTextSize(1);
    display.println(F("mg/dL"));
    display.display();
  #endif
}

void fillInputFromRing(TfLiteTensor* in) {
  if (!in) return;

  // Count elements in the input tensor
  int elem_count = 0;
  if (in->type == kTfLiteFloat32) elem_count = in->bytes / sizeof(float);
  else if (in->type == kTfLiteInt8) elem_count = in->bytes / sizeof(int8_t);
  else {
    Serial.print(F("ERROR: Unsupported input tensor type "));
    Serial.println(in->type);
    return;
  }

  // Expect at least WIN elements; if more, fill extra features with 0
  if (elem_count < WIN) {
    Serial.print(F("ERROR: Input elem_count < WIN. elem_count="));
    Serial.print(elem_count);
    Serial.print(F(" WIN="));
    Serial.println(WIN);
    return;
  }

  // Oldest sample index
  int start = (g_count == WIN) ? g_head : 0;

  // Helper: normalized 0..1 (since INPUT_IS_RAW_MGDL is 0 in current working mode)
  auto norm01 = [&](float mgdl) -> float {
    float x = (mgdl - MIN_G) / (MAX_G - MIN_G);
    return clampf(x, 0.0f, 1.0f);
  };

  // Fill the first WIN positions with the window, and the rest with 0
  if (in->type == kTfLiteFloat32) {
    float* dst = reinterpret_cast<float*>(in->data.raw);
    for (int i = 0; i < elem_count; i++) dst[i] = 0.0f;

    for (int i = 0; i < WIN; i++) {
      float mgdl = g_ring[(start + i) % WIN];
      dst[i] = norm01(mgdl);
    }
  } else { // int8
    int8_t* dst = reinterpret_cast<int8_t*>(in->data.raw);

    // Fill everything with the quantized value of 0.0 in real domain
    // For normalized input, "0.0" is valid
    int q0 = (int)lroundf(0.0f / in->params.scale) + in->params.zero_point;
    if (q0 < -128) q0 = -128;
    if (q0 > 127)  q0 = 127;
    for (int i = 0; i < elem_count; i++) dst[i] = (int8_t)q0;

    // write the window values into the first WIN elements
    for (int i = 0; i < WIN; i++) {
      float mgdl = g_ring[(start + i) % WIN];
      float v = norm01(mgdl);

      int q = (int)lroundf(v / in->params.scale) + in->params.zero_point;
      if (q < -128) q = -128;
      if (q > 127)  q = 127;
      dst[i] = (int8_t)q;
    }
  }
}



bool invokeAndReport() {
  const uint32_t t0 = micros();
  const TfLiteStatus st = interpreter->Invoke();
  const uint32_t t1 = micros();
  if (st != kTfLiteOk) {
    Serial.println(F("ERROR: Inference failed (Invoke)."));
    return false;
  }
  Serial.print(F("Invoke time (us): ")); Serial.println((int)(t1 - t0));
  printOutputSamples(output);
  return true;
}

void pushSample(float mgdl) {
  g_last_input_mgdl = mgdl;
  g_ring[g_head] = mgdl;
  g_head = (g_head + 1) % WIN;
  if (g_count < WIN) g_count++;

}

void predictFromRing() {
  if (g_count < WIN) {
    Serial.print(F("Buffered ")); Serial.print(g_count); Serial.print(F("/")); Serial.println(WIN);
    return;
  }
  fillInputFromRing(input);
  invokeAndReport();
}


// ===================== ANALOG SENSOR =====================
float mvToMgdl(float mv) {
  float mvc = clampf(mv, g_lo_mv, g_hi_mv);
  float frac = (mvc - g_lo_mv) / (g_hi_mv - g_lo_mv + 1e-9f);
  return frac * (g_hi_mgdl - g_lo_mgdl) + g_lo_mgdl;
}

float readSensorMv() {
  // ESP32 core helper (millivolts)
  int mv = analogReadMilliVolts(SENSOR_PIN);
  return (float)mv;

}

void sampleOnce() {
  float mv   = readSensorMv();
  float mgdl = mvToMgdl(mv);
  pushSample(mgdl);
  predictFromRing();
}

// ===================== SHELL / UI =====================
void printHelp() {
  Serial.println();
  Serial.println(F("Commands:"));
  Serial.println(F("  h                : help"));
  Serial.println(F("  s                : print input/output shapes"));
  Serial.println(F("  m                : print arena usage"));
  Serial.println(F("  t                : run one invoke on current input and show timing"));
  Serial.println(F("  p <mgdl>         : feed a FLAT window at mg/dL (overwrites ring) and predict"));
  Serial.println(F("  a <mgdl>         : ADD a single sample to rolling buffer (predicts when full)"));
  Serial.println(F("  r                : run quick suite (flat 90, 120, 150)"));
  Serial.println(F("  csv <v1,v2,...>  : play a short CSV of mg/dL values through the ring"));
  Serial.println(F("  bench            : flats & ramps; coarse MAE/RMSE vs last-sample proxy"));
  Serial.println(F("  start/stop       : start/stop analog sampling"));
  Serial.println(F("  rate <ms>        : set sampling period (e.g., rate 200)"));
  Serial.println(F("  mv               : print sensor millivolts"));
  Serial.println(F("  cal lo <mv> <mgdl>  : set low calibration point"));
  Serial.println(F("  cal hi <mv> <mgdl>  : set high calibration point"));
  Serial.println(F("  log on/off       : enable/disable CSV logging to SPIFFS"));
  Serial.println(F("  dump             : print the CSV file to Serial"));
  Serial.println(F("  clearlog         : delete the CSV file"));
  Serial.println();
}

String readLine() {
  static String line = "";
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (line.length() > 0) {
        String out = line; line = ""; return out;
      }
    } else {
      line += c;
    }
  }
  return String("");
}

void runQuickSuite() {
  float tests[] = {90.0f, 120.0f, 150.0f};
  for (float mgdl : tests) {
    Serial.print(F("\n[TEST] Flat window @ ")); Serial.print(mgdl); Serial.println(F(" mg/dL"));
    g_count = 0; g_head = 0;
    for (int i = 0; i < WIN; i++) pushSample(mgdl); predictFromRing();
    predictFromRing();
  }
}

void printWindow() {
  Serial.print(F("[WIN oldest→newest] "));
  int start = (g_count == WIN) ? (g_head % WIN) : 0;
  for (int i=0; i<g_count; i++) {
    float v = g_ring[(start + i) % WIN];
    Serial.print(v, 1);
    if (i < g_count-1) Serial.print(F(","));
  }
  Serial.println();
}


void runBench() {
  Serial.println(F("\n[Bench] Flats (90/120/150) and ramps (80→160 and 160→80)."));
  Stats s; resetStats(s);
  auto predict_and_accumulate = [&](float mgdl){
    pushSample(mgdl);
    predictFromRing();
    if (g_count >= WIN) {
      float y0_norm = dequantizeFirst(output);
      float y_pred  = y0_norm;
    #if OUTPUT_RANGE_ZERO_TO_ONE
      y_pred = y0_norm * (MAX_G - MIN_G) + MIN_G;
    #endif
      addSample(s, mgdl, y_pred); // crude proxy
    }
  };

  float flats[] = {90,120,150};
  for (float f : flats) { g_count = 0; g_head = 0; for (int i=0; i<WIN+6; i++) predict_and_accumulate(f); }

  g_count = 0; g_head = 0; for (float x=80;  x<=160; x+=8) predict_and_accumulate(x);
  g_count = 0; g_head = 0; for (float x=160; x>= 80; x-=8) predict_and_accumulate(x);

  finishStats(s);
  Serial.print(F("[Bench] N=")); Serial.print(s.n);
  Serial.print(F("  MAE=")); Serial.print(s.mae, 2);
  Serial.print(F(" mg/dL  RMSE=")); Serial.print(s.rmse, 2);
  Serial.println(F(" mg/dL"));
}

String sha1Hex(const char* input) {
  unsigned char out[20];
  mbedtls_sha1_context ctx;
  mbedtls_sha1_init(&ctx);
  mbedtls_sha1_starts(&ctx);
  mbedtls_sha1_update(&ctx, (const unsigned char*)input, strlen(input));
  mbedtls_sha1_finish(&ctx, out);
  mbedtls_sha1_free(&ctx);

  static const char* hex = "0123456789abcdef";
  char buf[41];
  for (int i = 0; i < 20; i++) {
    buf[i * 2] = hex[(out[i] >> 4) & 0xF];
    buf[i * 2 + 1] = hex[out[i] & 0xF];
  }
  buf[40] = '\0';
  return String(buf);
}

bool postPredictionToNightscout(float pred_mgdl, long long when_ms) {
  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient https;

  String url = String("https://") + NIGHTSCOUT_HOST + "/api/v1/entries.json";
  if (!https.begin(client, url)) return false;

  https.addHeader("Content-Type", "application/json");
  https.addHeader("api-secret", sha1Hex(API_SECRET_PLAINTEXT));

  DynamicJsonDocument doc(512);
  JsonArray arr = doc.to<JsonArray>();
  JsonObject obj = arr.createNestedObject();

 // obj["type"] = "sgv"; // Setting it as "sgv" shows the prediction as a regular grey dot on NightScout dashboard, hard to differentiate with regular dots from CGM
 obj["type"] = "mbg"; // Setting as "mbg" shows the prediction on NightScout dashboard as large red dot. Easier to differentiate from CGM reading dots
 obj["mbg"]  = (int)lroundf(pred_mgdl);
 // obj.remove("sgv");   // or just don’t set sgv at all

  obj["sgv"] = (int)lroundf(pred_mgdl);
  obj["date"] = when_ms;

  // (2) device label
  obj["device"] = "ESP32-PRED";

  //tooltip readability
  obj["noise"] = 1;

  obj["notes"] = "Pred +5m (ESP32)";


  String payload;
  serializeJson(arr, payload);

  int code = https.POST(payload);
  https.end();

  return (code == 200 || code == 201);
}

bool postPredictionNote(float pred_mgdl, long long when_ms) {
  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient https;

  String url = String("https://") + NIGHTSCOUT_HOST + "/api/v1/treatments.json";
  if (!https.begin(client, url)) return false;

  https.addHeader("Content-Type", "application/json");
  https.addHeader("api-secret", sha1Hex(API_SECRET_PLAINTEXT));

  DynamicJsonDocument doc(512);
  doc["eventType"]  = "Note";
  doc["created_at"] = millisToIso(when_ms);   
  doc["enteredBy"]  = "ESP32-PRED";
  doc["notes"]      = String("Pred: ") +
                      String((int)lroundf(pred_mgdl)) +
                      " mg/dL (+5 min forecast)";

  String payload;
  serializeJson(doc, payload);

  int code = https.POST(payload);
  https.end();

  return (code == 200 || code == 201);
}

String millisToIso(long long ms) {
  time_t sec = ms / 1000;
  struct tm t;
  gmtime_r(&sec, &t);

  char buf[30];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &t);
  return String(buf);
}

void syncTime() {
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  while (time(nullptr) < 1700000000) delay(500);
}

bool fetchLastSGV(int out[WIN], long long* latest_date_ms_out) {
  if (latest_date_ms_out) *latest_date_ms_out = 0;

  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient https;

  String url = String("https://") + NIGHTSCOUT_HOST +
               "/api/v1/entries/sgv.json?count=" + WIN;

  if (!https.begin(client, url)) return false;

  https.addHeader("api-secret", sha1Hex(API_SECRET_PLAINTEXT));
  if (https.GET() != 200) {
    https.end();
    return false;
  }

  DynamicJsonDocument doc(16 * 1024);
  DeserializationError err = deserializeJson(doc, https.getString());
  https.end();
  if (err) return false;

  JsonArray arr = doc.as<JsonArray>();
  if ((int)arr.size() < WIN) return false;

  // Nightscout returns newest to oldest.
  // Converting to oldest to newest for ring buffer.
  for (int i = 0; i < WIN; i++) {
    out[i] = arr[WIN - 1 - i]["sgv"].as<int>();
  }

  // newest entry is arr[0]
  long long latest_ms = 0;
  if (arr[0].containsKey("date")) {
    latest_ms = arr[0]["date"].as<long long>();
  } else if (arr[0].containsKey("mills")) {
    latest_ms = arr[0]["mills"].as<long long>();
  }
  if (latest_date_ms_out) *latest_date_ms_out = latest_ms;

  return true;
}


// ===================== SETUP/LOOP =====================
void setup() {
  Serial.begin(kUartBaud);
  delay(1000);
  Serial.println(F("\n[GlucoseBuddy] MVP: analog + TFLM + OLED + SPIFFS"));

  // SPIFFS
  if (!SPIFFS.begin(true)) { Serial.println(F("SPIFFS mount failed")); }
  else { Serial.println(F("SPIFFS mounted")); }

  // OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
  } else {
    #if USE_OLED
      display.clearDisplay();
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.println(F("GlucoseBuddy v1.0"));
      display.display();
      delay(500);
    #endif
  }

  // Analog init
  pinMode(SENSOR_PIN, INPUT);
  analogReadResolution(12);
  #ifdef ADC_11db
  analogSetPinAttenuation(SENSOR_PIN, ADC_11db);
  #endif

  // Register ops (MicroTFLite calls)
  resolver.AddUnpack();
  resolver.AddPack();
  resolver.AddConcatenation();
  resolver.AddSplit();
  resolver.AddStridedSlice();
  resolver.AddFullyConnected();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddReshape();
  resolver.AddLogistic();
  resolver.AddTanh();
  resolver.AddRelu();
  resolver.AddQuantize();
  resolver.AddDequantize();

  // Load model
  model = tflite::GetModel(glucose_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(F("Model schema mismatch. Model: "));
    Serial.print(model->version()); Serial.print(F("  TFLM: ")); Serial.println(TFLITE_SCHEMA_VERSION);
    haltWithMessage(F("ERROR: Incompatible model schema."));
  }
  Serial.print(F("Model bytes: ")); Serial.println(glucose_tflite_len);

  // Interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  Serial.println(F("Allocating tensors..."));
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  Serial.print(F("Allocate status = ")); Serial.println(alloc_status == kTfLiteOk ? F("OK") : F("FAIL"));
  if (alloc_status != kTfLiteOk) haltWithMessage(F("ERROR: AllocateTensors failed."));

  input  = interpreter->input(0);
  output = interpreter->output(0);

  printTensorShape("Input", input);
  printTensorShape("Output", output);
  Serial.print("Input bytes: "); Serial.println(input->bytes);
  Serial.print("Output bytes: "); Serial.println(output->bytes);
  Serial.print(F("TensorArena used/total (approx.): "));
  Serial.print(interpreter->arena_used_bytes()); Serial.print(F(" / ")); Serial.println(kTensorArenaSize);
  Serial.println("=== Quant params ===");
  Serial.print("input type: "); Serial.println(input->type);
  Serial.print("input scale: "); Serial.println(input->params.scale, 10);
  Serial.print("input zero_point: "); Serial.println(input->params.zero_point);

  Serial.print("output type: "); Serial.println(output->type);
  Serial.print("output scale: "); Serial.println(output->params.scale, 10);
  Serial.print("output zero_point: "); Serial.println(output->params.zero_point);


  printHelp();

  g_count = 0; g_head = 0;

  // ---- WiFi + time for Nightscout HTTPS ----
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  syncTime();

}


void handleCsvList(const String& line) {
  int sp = line.indexOf(' ');
  if (sp < 0 || sp + 1 >= line.length()) { Serial.println(F("Usage: csv v1,v2,...")); return; }

  String list = line.substring(sp + 1);
  list.replace(" ", "");  // remove spaces

  g_count = 0; g_head = 0;

  int start = 0;
  while (start < list.length()) {
    int comma = list.indexOf(',', start);
    String token = (comma == -1) ? list.substring(start) : list.substring(start, comma);
    token.trim();
    if (token.length() > 0) {
      float mgdl = token.toFloat();
      pushSample(mgdl);
    }
    if (comma == -1) break;
    start = comma + 1;
  }

  // auto-invoke once at end
  if (g_count >= WIN) {
    fillInputFromRing(input);
    invokeAndReport();
  } else {
    Serial.print(F("Buffered ")); Serial.print(g_count); Serial.print(F("/")); Serial.println(WIN);
  }

  Serial.println(F("[csv] Done."));
}


String readLine(); // forward

void loop() {
  // periodic sampler
  if (g_sampling && millis() >= g_next_ms) {
    g_next_ms = millis() + g_period_ms;
    sampleOnce();
  }

  String cmd = readLine();
  if (cmd.length() == 0) { delay(5); return; }
  cmd.trim();

  if (cmd == "h") {
    printHelp();
  } else if (cmd == "s") {
    printTensorShape("Input", input); printTensorShape("Output", output);
  } else if (cmd == "m") {
    Serial.print(F("Arena used: ")); Serial.println(interpreter->arena_used_bytes());
  } else if (cmd == "t") {
    invokeAndReport();
  } else if (cmd.startsWith("p ")) {
    float mgdl = cmd.substring(2).toFloat();
    Serial.print(F("Flat window at ")); Serial.print(mgdl); Serial.println(F(" mg/dL"));
    g_count = 0; g_head = 0; for (int i = 0; i < WIN; i++) pushSample(mgdl); predictFromRing();
  } else if (cmd.startsWith("a ")) {
    float mgdl = cmd.substring(2).toFloat();
    Serial.print(F("Add sample: ")); Serial.println(mgdl);
    pushSample(mgdl); predictFromRing();
  } else if (cmd == "r") {
    runQuickSuite();
  } else if (cmd.startsWith("csv")) {
    handleCsvList(cmd);
  } else if (cmd == "bench") {
    runBench();
  } else if (cmd == "start") {
    g_sampling = true; g_next_ms = millis(); Serial.println(F("Sampling: ON"));
  } else if (cmd == "stop") {
    g_sampling = false; Serial.println(F("Sampling: OFF"));
  } else if (cmd.startsWith("rate ")) {
    uint32_t v = (uint32_t)cmd.substring(5).toInt(); if (v < 50) v = 50;
    g_period_ms = v; Serial.print(F("Sampling period set to ")); Serial.print(g_period_ms); Serial.println(F(" ms"));
  } else if (cmd == "mv") {
    Serial.print(F("Sensor (mV): ")); Serial.println(readSensorMv(), 1);
  } else if (cmd.startsWith("cal lo ")) {
    int sp1 = cmd.indexOf(' ', 7);
    if (sp1 > 0) { g_lo_mv = cmd.substring(7, sp1).toFloat(); g_lo_mgdl = cmd.substring(sp1 + 1).toFloat();
      Serial.print(F("Low calibration: ")); Serial.print(g_lo_mv); Serial.print(F(" mV -> "));
      Serial.print(g_lo_mgdl); Serial.println(F(" mg/dL"));
    } else Serial.println(F("Usage: cal lo <mv> <mgdl>"));
  } else if (cmd.startsWith("cal hi ")) {
    int sp1 = cmd.indexOf(' ', 7);
    if (sp1 > 0) { g_hi_mv = cmd.substring(7, sp1).toFloat(); g_hi_mgdl = cmd.substring(sp1 + 1).toFloat();
      Serial.print(F("High calibration: ")); Serial.print(g_hi_mv); Serial.print(F(" mV -> "));
      Serial.print(g_hi_mgdl); Serial.println(F(" mg/dL"));
    } else Serial.println(F("Usage: cal hi <mv> <mgdl>"));
  } else if (cmd == "log on") {
    g_log_enabled = true;  Serial.println(F("[log] ON"));
  } else if (cmd == "log off") {
    g_log_enabled = false; Serial.println(F("[log] OFF"));
  } else if (cmd == "win") {
    printWindow();
  } else if (cmd == "dump") {
    File f = SPIFFS.open(kLogPath, FILE_READ);
    if (!f) Serial.println(F("[dump] no file"));
    else {
      Serial.println(F("---- /glucose_log.csv ----"));
      while (f.available()) Serial.write(f.read());
      f.close();
      Serial.println(F("---- end ----"));
    }
  } else if (cmd == "ns") {
  int sgv[WIN];
  long long latest_sgv_ms = 0;

  if (fetchLastSGV(sgv, &latest_sgv_ms)) {
    g_count = WIN;
    g_head  = 0;
    for (int i = 0; i < WIN; i++) g_ring[i] = (float)sgv[i];

    g_last_input_mgdl = (float)sgv[WIN - 1];  // newest for plotter/log

    Serial.print("[Nightscout] Latest SGV time (ms): ");
    Serial.println((long long)latest_sgv_ms);

    Serial.println("[Nightscout] Window loaded");
    predictFromRing();

    // Align prediction to latest SGV time + 5 minutes
    if (!isnan(g_last_pred_mgdl) && latest_sgv_ms > 0) {
      long long pred_time_ms = latest_sgv_ms + (60LL * 1000LL); 


      if (postPredictionToNightscout(g_last_pred_mgdl, pred_time_ms)){
        postPredictionNote(g_last_pred_mgdl, pred_time_ms + 10 * 1000); // Treatment note sent after a delay of 10 seconds so that it doesn't overlap with prediction note on Nightscout graph
        Serial.println("[Nightscout] Prediction + note posted");}
      else
        Serial.println("[Nightscout] Prediction POST failed");
    } else {
      Serial.println("[Nightscout] Skipped POST (no pred or bad time)");
    }

  } else {
    Serial.println("[Nightscout] Fetch failed");
  }
}
 else if (cmd == "clearlog") {
    if (SPIFFS.remove(kLogPath)) Serial.println(F("[log] cleared"));
    else                         Serial.println(F("[log] clear failed or not present"));
  } else {
    Serial.println(F("Unknown command. Type 'h' for help."));
  }
}
