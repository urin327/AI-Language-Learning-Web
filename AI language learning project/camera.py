import cv2
from ultralytics import YOLO
from deep_translator import GoogleTranslator

# YOLO model load
model = YOLO("yolov8n.pt")

# translator
translator = GoogleTranslator(source='en', target='ko')

# cache (хурд нэмнэ)
translated_cache = {}

# camera open
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    # draw results (box зураг)
    annotated_frame = results[0].plot()

    #  translation хэсэг
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]  # English

            # cache ашиглах
            if label in translated_cache:
                korean = translated_cache[label]
            else:
                korean = translator.translate(label)
                translated_cache[label] = korean

            print(label, "→", korean)

    # show
    cv2.imshow("YOLO Camera", annotated_frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()