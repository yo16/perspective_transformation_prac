import cv2
import numpy as np

from constants import COLOR_RANGE

# 画像から、ビリヤード台の四隅を返す
def get_corner_points(cv_image_original):
    # この関数内で加工するため、コピーする
    cv_image = cv_image_original.copy()

    # 輪郭を検出するための準備：前処理としてマスク画像を取得する
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(COLOR_RANGE["table"]["lower"])  # 青色のHSV範囲の下限
    upper_blue = np.array(COLOR_RANGE["table"]["upper"])  # 青色のHSV範囲の上限191
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    output_path = './tmp/corner_mask.jpg'
    cv2.imwrite(output_path, mask)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv_image_contours = cv_image_original.copy()
    #for cont in contours:
    #    print(cont)
    #    cv2.circle(cv_image_contours, (int(cont[0]), int(cont[1])), 1, (0,255,0), 1)
    cv2.drawContours(cv_image_contours, contours, -1, (0,0,0), 1)
    output_path = './tmp/table_contours.jpg'
    cv2.imwrite(output_path, cv_image_contours)
    

    # 点を左上、右上、右下、左下の順に並べ替える関数
    # 1. Y座標の小さな上位２つが上、大きな上位２つが下
    # 2. そのうち、X座標の小さな方が左、大きな方が右
    def order_points(pts):
        top_corners = pts[np.argsort(pts[:,1])][:2]
        bottom_corners = pts[np.argsort(pts[:,1])][2:]

        # 上位2点のうち、X座標が小さいものを左上とし、大きいものを右上とする
        tl = top_corners[np.argmin(top_corners[:, 0])]
        tr = top_corners[np.argmax(top_corners[:, 0])]

        # 下位2点のうち、X座標が小さいものを左下とし、大きいものを右下とする
        bl = bottom_corners[np.argmin(bottom_corners[:, 0])]
        br = bottom_corners[np.argmax(bottom_corners[:, 0])]

        # 左上、右上、右下、左下の順に並び替えた配列を返す
        return np.array([tl, tr, br, bl])

    # 最大輪郭を見つける
    largest_contour = max(contours, key=cv2.contourArea)

    # 輪郭の近似を行い、四角形の頂点を見つける
    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    assert len(approx) == 4, '角が４点見つかっていません！(' + str(len(approx)) + ')'
    # 近似した頂点が4つであれば、それらは四角形の角
    if len(approx) == 4:
        cv2.drawContours(cv_image, [approx], -1, (0, 255, 0), 1)  # 輪郭を緑色で描画
        approx = approx.reshape(4, 2)
        approx = order_points(approx)  # 頂点を並べ替える

    # # 結果を保存
    output_path = './tmp/table_with_corners.jpg'
    cv2.imwrite(output_path, cv_image)

    # 出力された角の座標
    return np.array(approx.tolist())