# 四隅を検出する

import cv2
import numpy as np

def detect_corners(image_path):
    # 画像を読み込む
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cannyエッジ検出器を用いてエッジを検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # ハフ変換を用いて直線を検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # 直線の交点を計算して四隅を求める
    corners = []
    if lines is not None:
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 他の全ての線と交差する点を求める
                for j in range(i+1, len(lines)):
                    for x3, y3, x4, y4 in lines[j]:
                        # 直線の方程式を計算
                        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        if denom == 0:
                            continue  # 平行線の場合
                        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                        # 画像の範囲内にある交点のみを保存
                        if 0 <= px <= image.shape[1] and 0 <= py <= image.shape[0]:
                            corners.append((px, py))

    # 画像に交点を描画
    for corner in corners:
        cv2.circle(image, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)

    # 結果の画像を表示
    cv2.imshow('Detected Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners

# 画像パスを指定して関数を呼び出す
image_path = './images/sample1.jpg'
corners = detect_corners(image_path)
print(corners)
