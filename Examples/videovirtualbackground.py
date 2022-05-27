import cv2
from virtualbg import SelfiSegmentation


def main():
    cap = cv2.VideoCapture(0)
    segmentor = SelfiSegmentation()
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        success, img = cap.read()
        if success:
            imgOut = segmentor.virtualBG(img, threshold=0.7)
            cv2.imshow("Image", img)
            cv2.imshow("Image Out", imgOut)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


if __name__ == "__main__":
    main()
