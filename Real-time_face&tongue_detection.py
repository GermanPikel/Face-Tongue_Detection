import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov11_tongue_face_v1.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            raise Exception("FRAME ERROR BRO")

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Face&Tongue detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
