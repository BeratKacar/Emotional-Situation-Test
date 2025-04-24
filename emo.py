import cv2
from deepface import DeepFace
import time

# Start the camera
cap = cv2.VideoCapture(0)

print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Attempt to analyze the imag
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for result in results:
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion]

            # Enclose the face in a square
            face_coordinates = result['region']
            x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Print the screen
            cv2.putText(frame, f'Duygu: {emotion} ({confidence:.2f}%)', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        print("Analiz hatası:", str(e))

    # Show image
    cv2.imshow("Duygu Analizi", frame)

    # Q has brake the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
