import cv2
import numpy as np

# 画像を読み込む
image_path = './images/sample1.jpg'
image = cv2.imread(image_path) # 実際にはここに画像のパスを指定します。

# 画像をHSV色空間に変換
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 青色の範囲を定義（HSV色空間で）
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# 青色の範囲に基づいたマスクを作成
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# マスクを元の画像に適用して青色のみを取得
blue_area = cv2.bitwise_and(image, image, mask=mask)

# マスクから輪郭を見つける
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 青色のエリアの輪郭を画像に描画（デバッグ用）
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

# 結果を保存する
cv2.imwrite('blue_table_with_contours.jpg', image_with_contours)
cv2.imwrite('blue_area.jpg', blue_area)
