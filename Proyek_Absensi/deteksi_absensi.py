import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
import numpy as np

# Folder tempat foto referensi disimpan
path_gambar = 'wajah_teman'
images = []
classNames = []
daftar_file = os.listdir(path_gambar)

print("Memuat foto referensi dan mengekstrak fitur wajah...")

# Membaca setiap gambar di folder wajah_teman
for cl in daftar_file:
    if cl.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(path_gambar, cl)
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is not None:
            images.append(img_bgr)
            # Mengambil nama dari nama file (misal: "Fadlan Lubis_depan.jpg" -> "Fadlan Lubis")
            nama_asli = os.path.splitext(cl)[0].split('_')[0]
            classNames.append(nama_asli)

# Fungsi untuk mendapatkan encoding wajah dari gambar referensi
def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img_rgb)[0]
            encodeList.append(encode)
        except IndexError:
            print("Wajah tidak terdeteksi dengan jelas pada salah satu gambar referensi.")
    return encodeList

encodeListKnown = findEncodings(images)
print(f"Selesai! {len(encodeListKnown)} data wajah dikenali. Memulai kamera...")

# Fungsi untuk mencatat ke Excel/CSV
def catat_absensi(nama):
    file_excel = 'Absensi.csv'
    
    # Buat file jika belum ada
    if not os.path.isfile(file_excel):
        df = pd.DataFrame(columns=['Nama', 'Tanggal', 'Waktu Hadir'])
        df.to_csv(file_excel, index=False)
        
    df = pd.read_csv(file_excel)
    
    sekarang = datetime.now()
    tanggal_hari_ini = sekarang.strftime('%Y-%m-%d')
    waktu_sekarang = sekarang.strftime('%H:%M:%S')
    
    # Cek apakah orang ini sudah absen hari ini agar tidak double-input
    sudah_absen = df[(df['Nama'] == nama) & (df['Tanggal'] == tanggal_hari_ini)]
    
    if sudah_absen.empty:
        df.loc[len(df)] = [nama, tanggal_hari_ini, waktu_sekarang]
        df.to_csv(file_excel, index=False)
        print(f"Berhasil mencatat kehadiran: {nama} pada {waktu_sekarang}")

# Membuka Kamera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
        
    # Perkecil ukuran frame agar proses deteksi lebih ringan dan real-time
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    # Deteksi lokasi wajah dan encodings pada frame saat ini
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Bandingkan wajah di kamera dengan data referensi
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            nama_terdeteksi = classNames[matchIndex].upper()
            
            # Kembalikan koordinat bounding box ke ukuran asli
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            
            # Gambar Bounding Box otomatis
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, nama_terdeteksi, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            
            # Panggil fungsi pencatatan
            catat_absensi(nama_terdeteksi)

    cv2.imshow('Sistem Absensi Deteksi Wajah', img)
    
    # Tekan 'ESC' untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()