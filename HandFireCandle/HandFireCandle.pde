/*
  Hand Fire -> Candle (Versi Processing / .pde)
  ================================================
  Deteksi tangan pakai deteksi WARNA KULIT + kontur (library OpenCV for
  Processing) - TIDAK butuh mediapipe/tensorflow, jadi bebas dari masalah
  instalasi seperti di Colab/Jupyter.

  Cara kerja:
  1. Kamera menangkap video.
  2. Setiap frame difilter berdasarkan warna kulit -> hasilnya gambar
     hitam-putih (mask) berisi area yang mirip warna kulit.
  3. Dicari kontur (blob) terbesar dari mask itu -> dianggap sebagai tangan.
  4. Titik TERATAS (topmost point) dari blob itu dipakai sebagai posisi
     "ujung jari" -> di situ partikel api di-spawn.
  5. Ada gambar lilin statis (PNG) ditampilkan tetap di posisi tertentu.
  6. Kalau titik ujung jari didekatkan ke sumbu lilin, lilin "menyala":
     partikel di tangan berhenti, partikel permanen muncul di sumbu lilin.

  ====================== INSTALASI (WAJIB DILAKUKAN DULU) ======================
  1. Buka Processing IDE (download di https://processing.org/download jika
     belum punya).
  2. Menu: Sketch > Import Library... > Add Library...
  3. Cari "OpenCV for Processing" (oleh Greg Borenstein) -> Install.
  4. Buat folder bernama "data" di dalam folder sketch ini (folder yang sama
     dengan file HandFireCandle.pde), lalu taruh gambar lilin kamu di situ
     dengan nama "candle.png". Idealnya gambar lilin yang BELUM menyala.
  5. Pastikan webcam tidak sedang dipakai aplikasi lain.
  6. Klik tombol Run (▶) di Processing IDE.

  Kontrol:
  - Tekan 'r' untuk memadamkan lilin lagi (reset).
  - Tekan 'd' untuk toggle tampilan debug (lihat mask deteksi kulit).
  - Tekan ESC / tutup jendela untuk keluar.
*/

import processing.video.*;
import gab.opencv.*;
import java.util.ArrayList;
import java.awt.Rectangle;

// ----------------------------------------------------------------------
// KONFIGURASI
// ----------------------------------------------------------------------

int CAM_W = 640;
int CAM_H = 480;

String CANDLE_IMAGE_PATH = "candle2.png";
float CANDLE_DISPLAY_WIDTH = 160;   // lebar tampilan lilin di layar (px)
float CANDLE_BOTTOM_MARGIN = 30;    // jarak lilin dari tepi bawah layar (px)
// Posisi X & Y dihitung otomatis di setup() supaya lilin ada di TENGAH BAWAH
float CANDLE_POS_X;
float CANDLE_POS_Y;

// Titik sumbu (wick) relatif terhadap gambar lilin (0.0 - 1.0)
float WICK_REL_X = 0.5;
float WICK_REL_Y = 0.12;

// Jarak (px) antara ujung jari & sumbu lilin supaya api "berpindah"
float IGNITE_DISTANCE = 55;

// Ambang minimal ukuran blob kulit supaya dianggap tangan (bukan noise)
float MIN_HAND_AREA = 1500;

int PARTICLES_PER_FRAME_HAND = 3;
int PARTICLES_PER_FRAME_CANDLE = 2;
int MAX_PARTICLES = 400;

// ----------------------------------------------------------------------
// VARIABEL GLOBAL
// ----------------------------------------------------------------------

Capture cam;
OpenCV opencv;
PImage candleImg;
PVector wickPoint;

ArrayList<FireParticle> handParticles = new ArrayList<FireParticle>();
ArrayList<FireParticle> candleParticles = new ArrayList<FireParticle>();

boolean candleLit = false;
boolean debugView = false;
float prevTime;

// Tombol "Ulangi" -> {x, y, w, h}
float[] resetButton;
boolean resetButtonHover = false;

// --- Deteksi wajah (untuk DIKECUALIKAN dari deteksi kulit tangan) ---
Rectangle[] lastFaces = new Rectangle[0];
int frameCounter = 0;
int FACE_DETECT_INTERVAL = 5;  // deteksi ulang wajah tiap 5 frame (hemat performa)

void setup() {
  size(640, 480);
  surface.setTitle("Hand Fire -> Candle (Processing)");

  // --- Setup kamera ---
  String[] cameras = Capture.list();
  if (cameras.length == 0) {
    println("Tidak ada kamera terdeteksi!");
    exit();
    return;
  }
  println("Kamera yang tersedia:");
  printArray(cameras);
  cam = new Capture(this, CAM_W, CAM_H);
  cam.start();

  // --- Setup OpenCV ---
  opencv = new OpenCV(this, CAM_W, CAM_H);
  opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE);

  // --- Load gambar lilin ---
  candleImg = loadImage(CANDLE_IMAGE_PATH);
  if (candleImg == null) {
    println("[PERINGATAN] '" + CANDLE_IMAGE_PATH + "' tidak ditemukan di folder data/.");
    println("Membuat placeholder lilin sederhana...");
    candleImg = createPlaceholderCandle();
  } else {
    float scale = CANDLE_DISPLAY_WIDTH / candleImg.width;
    candleImg.resize((int)CANDLE_DISPLAY_WIDTH, (int)(candleImg.height * scale));
  }

  // --- Posisikan lilin di TENGAH BAWAH layar ---
  CANDLE_POS_X = (width - candleImg.width) / 2.0f;
  CANDLE_POS_Y = height - candleImg.height - CANDLE_BOTTOM_MARGIN;

  wickPoint = new PVector(
    CANDLE_POS_X + candleImg.width * WICK_REL_X,
    CANDLE_POS_Y + candleImg.height * WICK_REL_Y
  );

  // --- Setup tombol "Ulangi" (posisi & ukuran) ---
  resetButton = new float[]{ width - 110, 16, 94, 32 }; // x, y, w, h

  prevTime = millis();
}

PImage createPlaceholderCandle() {
  int w = (int) CANDLE_DISPLAY_WIDTH;
  int h = (int) (w * 1.4);
  PGraphics pg = createGraphics(w, h);
  pg.beginDraw();
  pg.clear();
  pg.noStroke();
  pg.fill(235, 220, 150);
  pg.rect(w * 0.15f, h * 0.15f, w * 0.7f, h * 0.8f, 6);
  pg.fill(30);
  pg.rect(w * WICK_REL_X - 2, h * WICK_REL_Y - 10, 4, 20);
  pg.endDraw();
  PImage img = pg.get();
  return img;
}

void draw() {
  if (cam.available()) {
    cam.read();
  }

  float now = millis();
  float dt = (now - prevTime) / 1000.0;
  prevTime = now;

  // Tampilkan video (dimirror biar seperti cermin)
  pushMatrix();
  translate(width, 0);
  scale(-1, 1);
  image(cam, 0, 0, width, height);
  popMatrix();

  // --- Deteksi wajah secara berkala (supaya bisa dikecualikan dari deteksi tangan) ---
  PImage mirrored = getMirrored(cam);
  frameCounter++;
  if (frameCounter % FACE_DETECT_INTERVAL == 0) {
    lastFaces = detectFaces(mirrored);
  }

  // --- Deteksi tangan lewat warna kulit (area wajah dikecualikan) ---
  PVector fingerPos = detectHandTopmost(mirrored, lastFaces);

  // --- Spawn api HANYA di titik ujung jari yang terdeteksi ---
  if (fingerPos != null && !candleLit) {
    for (int i = 0; i < PARTICLES_PER_FRAME_HAND; i++) {
      if (handParticles.size() < MAX_PARTICLES) {
        handParticles.add(new FireParticle(fingerPos.x, fingerPos.y));
      }
    }
    float d = PVector.dist(fingerPos, wickPoint);
    if (d < IGNITE_DISTANCE) {
      candleLit = true;
      println("Lilin menyala! Api berpindah dari tangan ke lilin.");
    }

    // Penanda kecil di titik ujung jari, biar terlihat persis titik deteksinya
    noFill();
    stroke(0, 255, 0);
    strokeWeight(2);
    float m = 6;
    line(fingerPos.x - m, fingerPos.y, fingerPos.x + m, fingerPos.y);
    line(fingerPos.x, fingerPos.y - m, fingerPos.x, fingerPos.y + m);
  }

  // --- Gambar lilin statis ---
  image(candleImg, CANDLE_POS_X, CANDLE_POS_Y);

  // --- Update & gambar partikel tangan ---
  updateAndDraw(handParticles, dt);

  // --- Kalau lilin menyala, spawn api permanen di sumbu ---
  if (candleLit) {
    for (int i = 0; i < PARTICLES_PER_FRAME_CANDLE; i++) {
      if (candleParticles.size() < MAX_PARTICLES) {
        candleParticles.add(new FireParticle(wickPoint.x, wickPoint.y, 4, 1.2f, 2.2f));
      }
    }
  }
  updateAndDraw(candleParticles, dt);

  // --- Indikator jarak (bantu debug visual) ---
  if (fingerPos != null && !candleLit) {
    noFill();
    stroke(255, 255, 0);
    strokeWeight(1);
    ellipse(wickPoint.x, wickPoint.y, IGNITE_DISTANCE * 2, IGNITE_DISTANCE * 2);
    line(fingerPos.x, fingerPos.y, wickPoint.x, wickPoint.y);
  }

  // --- Debug: tampilkan mask deteksi kulit + area wajah yang dikecualikan ---
  if (debugView) {
    PImage mask = getSkinMask(mirrored);
    image(mask, width - 160, height - 120, 160, 120);
    noFill();
    stroke(255);
    rect(width - 160, height - 120, 160, 120);

    // Kotak biru = area wajah yang dikecualikan dari deteksi tangan
    noFill();
    stroke(0, 150, 255);
    strokeWeight(2);
    for (Rectangle face : lastFaces) {
      int marginX = (int) (face.width * 0.4);
      int marginTop = (int) (face.height * 0.6);
      int marginBottom = (int) (face.height * 0.9);
      rect(face.x - marginX, face.y - marginTop,
           face.width + marginX * 2, face.height + marginTop + marginBottom);
    }
  }

  // --- Status teks ---
  fill(255);
  noStroke();
  textSize(16);
  String status = candleLit ? "LILIN MENYALA" : "Dekatkan jari ke sumbu lilin";
  textAlign(CENTER);
  text(status, width / 2.0f, 30);
  textAlign(LEFT);

  // --- Tombol "Ulangi" ---
  drawResetButton();
}

void drawResetButton() {
  float bx = resetButton[0], by = resetButton[1], bw = resetButton[2], bh = resetButton[3];

  resetButtonHover = (mouseX >= bx && mouseX <= bx + bw && mouseY >= by && mouseY <= by + bh);

  noStroke();
  if (resetButtonHover) {
    fill(230, 90, 60);
  } else {
    fill(200, 60, 40);
  }
  rect(bx, by, bw, bh, 6);

  fill(255);
  textAlign(CENTER, CENTER);
  textSize(14);
  text("Ulangi", bx + bw / 2.0f, by + bh / 2.0f + 1);
  textAlign(LEFT, BASELINE);
}

// ----------------------------------------------------------------------
// DETEKSI TANGAN (warna kulit + kontur, ambil titik teratas)
// ----------------------------------------------------------------------

PImage getMirrored(PImage src) {
  PImage out = createImage(src.width, src.height, RGB);
  src.loadPixels();
  out.loadPixels();
  int w = src.width, h = src.height;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      out.pixels[y * w + (w - 1 - x)] = src.pixels[y * w + x];
    }
  }
  out.updatePixels();
  return out;
}

// Hasilkan mask hitam-putih: putih = area mirip kulit
PImage getSkinMask(PImage src) {
  PImage mask = createImage(src.width, src.height, RGB);
  src.loadPixels();
  mask.loadPixels();
  for (int i = 0; i < src.pixels.length; i++) {
    color c = src.pixels[i];
    float r = red(c), g = green(c), b = blue(c);
    boolean isSkin = isSkinColor(r, g, b);
    mask.pixels[i] = isSkin ? color(255) : color(0);
  }
  mask.updatePixels();
  return mask;
}

// Aturan klasik deteksi warna kulit berbasis RGB
boolean isSkinColor(float r, float g, float b) {
  float maxC = max(r, max(g, b));
  float minC = min(r, min(g, b));
  return (r > 95 && g > 40 && b > 20 &&
          (maxC - minC) > 15 &&
          abs(r - g) > 15 &&
          r > g && r > b);
}

// Deteksi wajah pakai Haar Cascade (buat DIKECUALIKAN dari deteksi tangan)
Rectangle[] detectFaces(PImage mirroredCam) {
  opencv.loadImage(mirroredCam);
  return opencv.detect();
}

// Hitamkan sebuah area persegi di dalam mask (dipakai untuk "menghapus" area
// wajah dari mask kulit supaya tidak ikut dianggap sebagai tangan)
void blackoutRect(PImage img, int rx, int ry, int rw, int rh) {
  img.loadPixels();
  int w = img.width, h = img.height;
  int x0 = max(0, rx);
  int y0 = max(0, ry);
  int x1 = min(w, rx + rw);
  int y1 = min(h, ry + rh);
  for (int y = y0; y < y1; y++) {
    for (int x = x0; x < x1; x++) {
      img.pixels[y * w + x] = color(0);
    }
  }
  img.updatePixels();
}

// Cari blob TANGAN via OpenCV (area wajah sudah dikecualikan), kembalikan
// titik teratasnya sebagai perkiraan posisi ujung jari.
//
// CIRI KHAS TANGAN yang dipakai untuk membedakannya dari blob kulit lain
// (leher, telinga, dsb) yang mungkin lolos dari exclusion wajah:
//   Tangan yang diangkat ke kamera SELALU tersambung ke lengan yang keluar
//   dari tepi frame (bawah/kiri/kanan) - beda dengan kepala yang biasanya
//   "mengambang" di tengah frame tanpa menyentuh tepi.
//   -> Kita prioritaskan blob yang bounding box-nya menyentuh salah satu
//      tepi tersebut sebagai kandidat tangan.
PVector detectHandTopmost(PImage mirroredCam, Rectangle[] faces) {
  PImage mask = getSkinMask(mirroredCam);

  // --- Kecualikan area wajah (+ perkiraan rambut/dahi & leher) dari mask ---
  // supaya wajah/kepala tidak ikut dianggap sebagai "tangan"
  for (Rectangle face : faces) {
    int marginX = (int) (face.width * 0.4);
    int marginTop = (int) (face.height * 0.6);    // tutupi dahi & rambut di atas wajah
    int marginBottom = (int) (face.height * 0.9); // tutupi leher di bawah wajah
    int rx = face.x - marginX;
    int ry = face.y - marginTop;
    int rw = face.width + marginX * 2;
    int rh = face.height + marginTop + marginBottom;
    blackoutRect(mask, rx, ry, rw, rh);
  }

  opencv.loadImage(mask);
  opencv.gray();
  opencv.threshold(80);
  opencv.dilate();
  opencv.erode();

  ArrayList<Contour> contours = opencv.findContours();

  int EDGE_MARGIN = 20; // toleransi (px) dianggap "menyentuh tepi frame"

  // --- Prioritas 1: blob yang menyentuh tepi frame (bawah/kiri/kanan) ---
  // -> ini kandidat paling kuat sebagai tangan+lengan
  Contour handContour = null;
  float handArea = 0;
  for (Contour c : contours) {
    float area = c.area();
    if (area < MIN_HAND_AREA) continue;

    Rectangle bbox = c.getBoundingBox();
    boolean touchesLeft = bbox.x <= EDGE_MARGIN;
    boolean touchesRight = (bbox.x + bbox.width) >= (mask.width - EDGE_MARGIN);
    boolean touchesBottom = (bbox.y + bbox.height) >= (mask.height - EDGE_MARGIN);

    if ((touchesLeft || touchesRight || touchesBottom) && area > handArea) {
      handArea = area;
      handContour = c;
    }
  }

  // --- Fallback: kalau tidak ada blob yang menyentuh tepi, pakai blob
  //     terbesar apa adanya (lebih baik dari tidak ada deteksi sama sekali) ---
  if (handContour == null) {
    for (Contour c : contours) {
      float area = c.area();
      if (area > handArea) {
        handArea = area;
        handContour = c;
      }
    }
  }

  if (handContour == null || handArea < MIN_HAND_AREA) {
    return null;
  }

  ArrayList<PVector> pts = handContour.getPoints();

  // Cari titik Y paling atas (paling ekstrem/ujung) dari kontur tangan
  float minY = Float.MAX_VALUE;
  for (PVector p : pts) {
    if (p.y < minY) minY = p.y;
  }

  // Rata-ratakan semua titik yang dekat dengan titik paling atas (toleransi 6px)
  // -> ini mengurangi noise/jitter dan lebih presisi menunjuk ujung jari asli,
  //    dibanding cuma mengambil 1 titik piksel mentah yang bisa melompat-lompat.
  float sumX = 0, sumY = 0;
  int count = 0;
  float TOLERANCE = 6;
  for (PVector p : pts) {
    if (p.y <= minY + TOLERANCE) {
      sumX += p.x;
      sumY += p.y;
      count++;
    }
  }

  if (count == 0) return null;
  return new PVector(sumX / count, sumY / count);
}

// ----------------------------------------------------------------------
// SISTEM PARTIKEL API
// ----------------------------------------------------------------------

class FireParticle {
  float x, y;
  float vx, vy;
  float life, maxLife;
  float sz;
  float wobble;

  FireParticle(float px, float py) {
    this(px, py, 4, 1.5f, 3.0f);  // spread diperkecil supaya nempel di titik jari
  }

  FireParticle(float px, float py, float spread, float minUp, float maxUp) {
    x = px + random(-spread, spread);
    y = py + random(-spread * 0.3f, spread * 0.3f);
    vx = random(-0.5f, 0.5f);
    vy = -random(minUp, maxUp);
    maxLife = random(0.4f, 0.9f);
    life = maxLife;
    sz = random(6, 14);
    wobble = random(0, TWO_PI);
  }

  boolean update(float dt) {
    wobble += dt * 10;
    x += vx + sin(wobble) * 0.6f;
    y += vy;
    vy -= 0.02f;
    life -= dt;
    return life > 0;
  }

  void display() {
    float t = constrain(life / maxLife, 0, 1);
    float size = sz * t;
    if (size < 1) return;

    color col;
    if (t > 0.66f) {
      col = color(255, 200, 60);       // kuning terang
    } else if (t > 0.33f) {
      col = color(255, 120, 20);       // oranye
    } else {
      col = color(180, 40, 10);        // merah gelap
    }

    noStroke();
    float alpha = (0.55f * t + 0.15f) * 255;
    fill(red(col), green(col), blue(col), alpha);
    ellipse(x, y, size * 2, size * 2);

    fill(255, 255, 200, alpha);
    float coreSize = max(1, size / 3);
    ellipse(x, y - size * 0.2f, coreSize * 2, coreSize * 2);
  }
}

void updateAndDraw(ArrayList<FireParticle> list, float dt) {
  for (int i = list.size() - 1; i >= 0; i--) {
    FireParticle p = list.get(i);
    if (p.update(dt)) {
      p.display();
    } else {
      list.remove(i);
    }
  }
}

// ----------------------------------------------------------------------
// INPUT KEYBOARD
// ----------------------------------------------------------------------

void keyPressed() {
  if (key == 'r' || key == 'R') {
    resetCandle();
  } else if (key == 'd' || key == 'D') {
    debugView = !debugView;
  } else if (key == ESC) {
    // biarkan default (keluar)
  }
}

void mousePressed() {
  float bx = resetButton[0], by = resetButton[1], bw = resetButton[2], bh = resetButton[3];
  if (mouseX >= bx && mouseX <= bx + bw && mouseY >= by && mouseY <= by + bh) {
    resetCandle();
  }
}

// Fungsi reset terpusat -> dipanggil dari tombol maupun tombol keyboard 'r'
void resetCandle() {
  candleLit = false;
  candleParticles.clear();
  handParticles.clear();
  println("Lilin dipadamkan kembali / direset.");
}
