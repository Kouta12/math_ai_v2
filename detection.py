import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
from imutils import contours

classes= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', '=']
num_classes = len(classes)
image_size = 224


# 画像の読み込み
read_file = "./test_images/ccr_test_01.jpg"
image = cv2.imread(read_file, cv2.IMREAD_GRAYSCALE)


# 白黒反転
image = 255 - image

# 輪郭抽出
cnts, hierarch = cv2.findContours(
    image,
    cv2.RETR_EXTERNAL,       # 一番外側の輪郭のみを取得
    cv2.CHAIN_APPROX_SIMPLE    # 縦、横、斜め45°方向に完全に直線の部分の輪郭の点を省略します。
)
cnts, hierarchy = contours.sort_contours(cnts, method='left-to-right')

# 画像表示
img_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# ブロック検出：文字領域検出した輪郭の「横幅」が、以下の範囲なら輪郭を残す
block_horizontal_height_minimum = 5  
block_horizontal_height_max = 1000  

# ブロック検出：文字領域検出した輪郭の「縦の高さ」が、以下の範囲なら輪郭を残す
block_vertical_height_minimum = 5  
block_vertical_height_max = 1000   

text = []
index = 0

# 輪郭上の点の描写
for i, contour in enumerate(cnts):
    
    # 傾いていない外接する短径領域
    x, y, w, h = cv2.boundingRect(contour)
        
    if  not block_vertical_height_minimum < w < block_vertical_height_max:
      continue
    if  not block_horizontal_height_minimum < h < block_horizontal_height_max:
      continue    

    image_cropping = image[y:y+h-1, x:x+w-1]
    
    
    img = cv2.resize(image_cropping, dsize=(image_size, image_size))
    data = np.asarray(img) / 255
    data = np.expand_dims(data, -1)
    X = []
    X.append(data)
    X = np.array(X)

    model = load_model('./scr_model/vgg16_transfer_02.h5')

    result = model.predict(X)[0]
    predicted = result.argmax()
    # print(classes[predicted])
    text.append(classes[predicted])
    cv2.imwrite('./cropping_images/cropping_img_{}.jpg'.format(index), img)
    index += 1

    cv2.putText(img_disp, text="{}".format(classes[predicted]), org=(int(x+w/2), y+h+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(0,0,255))
    cv2.rectangle(img_disp, (x,y), (x+w, y+h), (0,255,0), 2)


print(*text)


# 画像の表示

cv2.imshow('image', img_disp)
cv2.waitKey(0)