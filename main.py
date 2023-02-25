from PIL import ImageFont, Image, ImageDraw
import numpy as np
# import easyocr
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# setx TESSDATA_PREFIX "D:\\Pro\\pycharm\\tesseract"
# Param

# loading images and resizing
img = cv2.imread("car4.jpg")
img = cv2.resize(img, (800, 600))

# load font
fontpath = './arial.ttf'
font = ImageFont.truetype(fontpath, 32)
b, g, r, a = 0, 255, 0, 0

# making the image grayscale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approximation = cv2.approxPolyDP(c, 0.02*perimeter, True)
    print(approximation)
    if len(approximation) == 4:
        number_plate_shape = approximation
        cv2.drawContours(img, [number_plate_shape], -1, (0, 255, 0), 3)  # draw a green square on the image
        break


count = 0
idx = 7

for c in contours:
# approximate the license plate contour
   contour_perimeter = cv2.arcLength(c, True)
   approx = cv2.approxPolyDP(c, 0.018 * contour_perimeter, True)
# Look for contours with 4 corners
   if len(approx) == 4:
     screenCnt = approx
# find the coordinates of the license plate contour
     x, y, w, h = cv2.boundingRect(c)
     new_img = img [ y: y + h, x: x + w]
# stores the new image
     cv2.imwrite('./'+str(idx)+'.jpg',new_img)
     idx += 1
     break

# draws the license plate contour on original image
# cv2.drawContours(original_image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("detected license plate", img)
# filename of the cropped license plate image
cropped_License_Plate = './7.jpg'
cv2.imshow("cropped license plate", cv2.imread(cropped_License_Plate))

# display the image with the square
# cv2.imshow("Number plate detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# converts the license plate characters to string
text = pytesseract.image_to_string(cropped_License_Plate, lang='eng' ,config='--psm 11')
parent = np.array(['0', '1', '2', '3', '4','5', '6', '7', '8', '9','-', '.',
                   'Q','W', 'E', 'R', 'T', 'Y','U', 'I', 'O', 'P', 'A','S', 'D',
                   'F', 'G', 'H', 'J', 'K','L', 'Z', 'X', 'C', 'V','B', 'N', 'M'])
arrList = list(text)
print(arrList)
for i in arrList:
    if i not in parent:
        arrList.remove(i)
myResult  = "".join(arrList)
print("License plate is: ", myResult)
cv2.waitKey(0)
cv2.destroyAllWindows()
