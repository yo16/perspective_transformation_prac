import cv2
import numpy as np
from enum import Enum

class Color(Enum):
    RED = 1
    YELLOW = 2
    WHITE = 3

def detect_balls(image_path, vertices, color):
    # 画像の読み込み
    image = cv2.imread(image_path)
    image_forEdit = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if (color == Color.RED):
        # 赤色のHSV範囲
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 100, 50])
        upper_red2 = np.array([180, 255, 255])

        # 赤色は、HSVで２か所で出るため、ORで結合する
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        cv2.imwrite('./tmp/detect_ball_red_mask1.jpg', mask1)
        cv2.imwrite('./tmp/detect_ball_red_mask2.jpg', mask2)

    elif (color == Color.YELLOW):
        # 黄色のHSV範囲
        # lower_yellow = np.array([15, 100, 100])
        # upper_yellow = np.array([30, 255, 255])
        lower_yellow = np.array([15, 0, 0])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        print("yellow!")
    else:
        # 白色のHSV範囲
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 70, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imwrite('./tmp/detect_ball_mask.jpg', mask)


    # 指定された四角形領域でマスクを適用
    #cv2.fillPoly(mask, [vertices], 0)
    #cv2.drawContours(mask, [vertices], -1, 255, -1)
    rect_mask = np.zeros_like(mask)
    cv2.fillPoly(rect_mask, [vertices], 255)  # 矩形内部を255で塗りつぶす
    masked_image = cv2.bitwise_and(mask, rect_mask)  # 元のマスクと矩形マスクのANDを取る

    cv2.imwrite('./tmp/detect_ball_rect.jpg', masked_image)

    # オープニング（縮小して拡大、ノイズ除去）
    kernel1 = np.ones((5,5), np.uint8)
    opening_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel1)
    cv2.imwrite('./tmp/detect_ball_open.jpg', opening_image)

    # クロージング（拡大して縮小、穴埋め）
    kernel2 = np.ones((5,5), np.uint8)
    closing_image = cv2.morphologyEx(opening_image, cv2.MORPH_CLOSE, kernel2)
    cv2.imwrite('./tmp/detect_ball_close.jpg', closing_image)


    # 球の検出
    balls = []
    contours, _ = cv2.findContours(closing_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:  # 最小サイズのしきい値
            balls.append((int(x), int(y), int(radius)))
            cv2.circle(image_forEdit, (int(x), int(y)), int(radius), (0,255,0), 1)
    ## 輪郭と階層情報を取得
    #contours, hierarchy = cv2.findContours(masked_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #circularity_threshold = 0.2
    #min_radius = 5
    #for i, contour in enumerate(contours):
    #    if hierarchy[0, i, 3] == -1:  # 他の輪郭に囲まれていない輪郭のみを考慮
    #        area = cv2.contourArea(contour)
    #        perimeter = cv2.arcLength(contour, True)
    #        if perimeter == 0:
    #            continue
    #            
    #        circularity = (4 * np.pi * area) / (perimeter ** 2)
    #        if circularity > circularity_threshold:
    #            ((x, y), radius) = cv2.minEnclosingCircle(contour)
    #            if radius > min_radius:
    #                cv2.circle(image_forEdit, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    #                balls.append((int(x), int(y), int(radius)))

    output_path = './tmp/found_balls.jpg'
    cv2.imwrite(output_path, image_forEdit)

    return balls


# verticesは四角形の頂点の座標のnumpy配列
corners = [[620, 398], [1012, 415], [846, 717], [103, 609]]
vertices = np.array(corners)

image_path = './images/sample1.jpg'
balls = detect_balls(image_path, vertices, Color.RED)

print(balls)
