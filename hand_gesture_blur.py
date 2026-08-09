"""
Hand Gesture Blur Camera (Custom Gesture Version)
---------------------------------------------------
Skenario:
- Kamera menyala normal, BELUM ada gesture yang tersimpan.
- Tekan tombol '1' -> gesture tangan yang sedang kamu tunjukkan SAAT ITU akan
  disimpan sebagai "gesture target".
- Setelah tersimpan, kamera hanya akan BLUR ketika tangan kamu membentuk
  gesture yang sama persis dengan yang disimpan tadi. Kalau bentuk tangan
  berbeda (atau tidak ada tangan), kamera kembali normal (tidak blur).
- Tekan '1' lagi kapan saja untuk mengganti/menyimpan ulang gesture target.
- Tekan 'r' untuk menghapus gesture yang tersimpan (reset).
- Tekan 'q' untuk keluar.

Requirement:
    pip install opencv-python mediapipe
"""

import cv2
import mediapipe as mp
import numpy as np

# ---------- Setup MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

WRIST = 0
MIDDLE_MCP = 9  # dipakai sebagai referensi skala tangan

# Seberapa mirip gesture saat ini harus dengan gesture tersimpan agar dianggap
# "match". Semakin KECIL nilainya, semakin ketat (harus makin persis mirip).
MATCH_THRESHOLD = 0.15


def landmarks_to_vector(landmarks):
    """
    Ubah 21 landmark tangan jadi vektor fitur yang:
    - Invariant terhadap posisi tangan di frame (dinormalisasi relatif ke wrist)
    - Invariant terhadap ukuran tangan / jarak ke kamera (dinormalisasi dengan
      skala tangan)
    Sehingga gesture yang sama akan menghasilkan vektor yang mirip walau
    posisi/tangan sedikit bergeser atau maju-mundur dari kamera.
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks])

    wrist = pts[WRIST]
    pts = pts - wrist  # relatif ke pergelangan tangan

    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale  # normalisasi skala

    return pts.flatten()


def gesture_distance(vec_a, vec_b):
    """Rata-rata jarak Euclidean antar titik landmark yang berpadanan."""
    a = vec_a.reshape(-1, 2)
    b = vec_b.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa membuka kamera. Pastikan kamera terhubung dan tidak dipakai aplikasi lain.")
        return

    saved_gesture = None   # vektor fitur gesture target, None = belum ada
    blur_active = False
    current_distance = None

    print("Kamera aktif.")
    print("Tekan '1' untuk menyimpan gesture tangan saat ini sebagai target.")
    print("Tekan 'r' untuk reset (hapus gesture tersimpan).")
    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)  # mirror biar lebih natural
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        current_vector = None
        blur_active = False  # default tiap frame: tidak blur, kecuali match

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_vector = landmarks_to_vector(hand_landmarks.landmark)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            if saved_gesture is not None and current_vector is not None:
                current_distance = gesture_distance(current_vector, saved_gesture)
                if current_distance < MATCH_THRESHOLD:
                    blur_active = True
        else:
            current_distance = None

        # ---------- Terapkan blur jika gesture match ----------
        if blur_active:
            frame = cv2.GaussianBlur(frame, (55, 55), 0)

        # ---------- Overlay info ----------
        if saved_gesture is None:
            gesture_status = "Belum ada gesture tersimpan (tekan '1')"
            gcolor = (0, 255, 255)
        else:
            gesture_status = "Gesture tersimpan"
            gcolor = (0, 255, 0)

        blur_status = "BLUR ON" if blur_active else "BLUR OFF"
        bcolor = (0, 0, 255) if blur_active else (0, 255, 0)

        cv2.putText(frame, gesture_status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, gcolor, 2)
        cv2.putText(frame, f"Status: {blur_status}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, bcolor, 2)
        if current_distance is not None:
            cv2.putText(frame, f"Distance: {current_distance:.3f} (threshold {MATCH_THRESHOLD})",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "[1] simpan gesture  [r] reset  [q] keluar",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Hand Gesture Blur Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            if current_vector is not None:
                saved_gesture = current_vector.copy()
                print("Gesture berhasil disimpan sebagai target baru.")
            else:
                print("Tidak ada tangan terdeteksi, gesture belum disimpan.")
        elif key == ord('r'):
            saved_gesture = None
            print("Gesture tersimpan telah direset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
