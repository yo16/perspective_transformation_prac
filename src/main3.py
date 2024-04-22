import cv2
import numpy as np
from enum import Enum


# 画像から、ビリヤード台の四隅を返す
def get_corner_points(cv_image_original):
    # この関数内で加工するため、コピーする
    cv_image = cv_image_original.copy()

    # 輪郭を検出するための準備：前処理としてマスク画像を取得する
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])  # 青色のHSV範囲の下限
    upper_blue = np.array([140, 255, 255])  # 青色のHSV範囲の上限
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    # 近似した頂点が4つであれば、それらは四角形の角
    if len(approx) == 4:
        cv2.drawContours(cv_image, [approx], -1, (0, 255, 0), 5)  # 輪郭を緑色で描画
        approx = approx.reshape(4, 2)
        approx = order_points(approx)  # 頂点を並べ替える

    # # 結果を保存
    output_path = './tmp/table_with_corners.jpg'
    cv2.imwrite(output_path, cv_image)

    # 出力された角の座標
    return np.array(approx.tolist())


# ボールの位置を取得
class Color(Enum):
    RED = 1
    YELLOW = 2
    WHITE = 3
def get_ball_positions(cv_image_original, vertices):

    def get_ball_position_by_color(cv_image, vertices, color):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
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
        #print(vertices)
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
        
        return np.array(balls)
    
    # この関数内で加工するため、コピーする
    cv_image = cv_image_original.copy()

    # 赤、白、黄色のボールの位置を取得
    red_balls = get_ball_position_by_color(cv_image, vertices, Color.RED)
    white_balls = get_ball_position_by_color(cv_image, vertices, Color.WHITE)
    yellow_balls = get_ball_position_by_color(cv_image, vertices, Color.YELLOW)

    # 位置を描画
    for ball in red_balls:
        cv2.circle(cv_image, (int(ball[0]), int(ball[1])), int(ball[2]), (0,255,0), 1)
        cv2.putText(cv_image,"red", (ball[0], ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, 2)
    for ball in white_balls:
        cv2.circle(cv_image, (int(ball[0]), int(ball[1])), int(ball[2]), (0,255,0), 1)
        cv2.putText(cv_image,"white", (ball[0], ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, 2)
    for ball in yellow_balls:
        cv2.circle(cv_image, (int(ball[0]), int(ball[1])), int(ball[2]), (0,255,0), 1)
        cv2.putText(cv_image,"yellow", (ball[0], ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, 2)
    
    output_path = './tmp/balls.jpg'
    cv2.imwrite(output_path, cv_image)

    # 各色でRの最大のものを採用する
    rwy_balls = [
        red_balls[np.argmax(red_balls[:,2])].tolist(),
        white_balls[np.argmax(white_balls[:,2])].tolist(),
        yellow_balls[np.argmax(yellow_balls[:,2])].tolist()
    ]
    print(rwy_balls)

    return rwy_balls


# perspective transformationで長方形へ変換
def cut_and_transform(original_image, points, width, height, balls, ball_size):
    src_points = np.float32(points)
    print(src_points)

    # 真上から見たビリヤード台の四隅の座標 (x, y) (通常は四角形)
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # ホモグラフィ行列を計算する
    matrix = cv2.getPerspectiveTransform(src_points, np.array(dst_points))
    matrix = np.float32(matrix)
    print(matrix)

    # 透視変換を実行する
    warped_image = cv2.warpPerspective(original_image, matrix, (width, height))

    # ボール座標をperspective transformationして、絵を描く
    
    # 元の点のリスト (x, y) 形式
    points = [(x, y) for x, y, _ in balls]
    # 同次座標に変換
    points_homogeneous = np.array([list(point) + [1] for point in points])
    # ホモグラフィ行列による変換
    transformed_points_homogeneous = np.dot(points_homogeneous, matrix.T)  # 行列を転置して乗算
    # 同次座標から通常の座標への変換
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, [2]]
    # 円を描く
    rwy_colors = ((0,0,255), (255,255,255), (0,255,255))
    for i, point in enumerate(transformed_points):
        cv2.circle(warped_image, (int(point[0]), int(point[1])), int(ball_size), rwy_colors[i], -1)

    # 結果の画像を表示する
    cv2.imshow('Warped Image', warped_image)

    # キー入力を待つ (これはWindowsのGUIを使用する場合)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 結果の画像をファイルに保存する
    cv2.imwrite('./tmp/warped_image.png', warped_image)



def main():
    # 画像を読み込む
    image_path = './images/sample1.jpg'
    image_scale = 2.0
    width = int(160 * image_scale)
    height = int(290 * image_scale)
    ball_size = 6.15 * image_scale

    original_image = cv2.imread(image_path)

    # ビリヤード台の角を取得
    corners = get_corner_points(original_image)

    # ボールの位置を取得
    rwy_balls = get_ball_positions(original_image, corners)
    print(11)
    print(rwy_balls)

    # perspective transformationで長方形へ変換
    cut_and_transform(original_image, corners, width, height, rwy_balls, ball_size)



if __name__=="__main__":
    main()
