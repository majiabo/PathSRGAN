import cv2

if __name__ == '__main__':
    size = 512
    img = cv2.imread('./images/1037.tif')
    cv2.imwrite('./images/1037.png', cv2.resize(img, None, fx=0.5, fy=0.5))