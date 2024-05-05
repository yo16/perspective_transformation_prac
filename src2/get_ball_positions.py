import cv2
import numpy as np

from constants import Color, COLOR_RANGE

# 画像の中の、ボールの位置を返す（ついでにRも）
def get_ball_positions(cv_image_original, vertices):

    def get_ball_position_by_color(cv_image, vertices, color):
        color_str = 'red' if (color==Color.RED) else 'yellow' if (color==Color.YELLOW) else 'white'

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = None
        if (color == Color.RED):
            # 赤色のHSV範囲
            lower_red1 = np.array(COLOR_RANGE["red"][0]["lower"])
            upper_red1 = np.array(COLOR_RANGE["red"][0]["upper"])
            lower_red2 = np.array(COLOR_RANGE["red"][1]["lower"])
            upper_red2 = np.array(COLOR_RANGE["red"][1]["upper"])

            # 赤色は、HSVで２か所で出るため、ORで結合する
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            cv2.imwrite('./tmp/detect_ball_red_mask1.jpg', mask1)
            cv2.imwrite('./tmp/detect_ball_red_mask2.jpg', mask2)
            cv2.imwrite('./tmp/detect_ball_red_mask3.jpg', mask)

        elif (color == Color.YELLOW):
            # 黄色のHSV範囲
            lower_yellow = np.array(COLOR_RANGE["yellow"]["lower"])
            upper_yellow = np.array(COLOR_RANGE["yellow"]["upper"])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            cv2.imwrite('./tmp/detect_ball_yellow_mask.jpg', mask)
        else:
            # 白色のHSV範囲
            lower_white = np.array(COLOR_RANGE["white"]["lower"])
            upper_white = np.array(COLOR_RANGE["white"]["upper"])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            cv2.imwrite('./tmp/detect_ball_white_mask.jpg', mask)

        # 指定された四角形領域でマスクを適用
        #cv2.fillPoly(mask, [vertices], 0)
        #cv2.drawContours(mask, [vertices], -1, 255, -1)
        rect_mask = np.zeros_like(mask)
        #print(vertices)
        cv2.fillPoly(rect_mask, [vertices], 255)  # 矩形内部を255で塗りつぶす
        masked_image = cv2.bitwise_and(mask, rect_mask)  # 元のマスクと矩形マスクのANDを取る

        cv2.imwrite('./tmp/detect_ball_rect_'+color_str+'.jpg', masked_image)

        # オープニング（縮小して拡大、ノイズ除去）
        kernel1 = np.ones((5,5), np.uint8)
        opening_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, kernel1)
        cv2.imwrite('./tmp/detect_ball_open_'+color_str+'.jpg', opening_image)

        # クロージング（拡大して縮小、穴埋め）
        kernel2 = np.ones((5,5), np.uint8)
        closing_image = cv2.morphologyEx(opening_image, cv2.MORPH_CLOSE, kernel2)
        cv2.imwrite('./tmp/detect_ball_close_'+color_str+'.jpg', closing_image)

        # 球の検出
        balls = []
        contours, _ = cv2.findContours(closing_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            #print(radius)
            if radius > 2:  # 最小サイズのしきい値
                balls.append((int(x), int(y), int(radius)))
        
        return np.array(balls)
    
    # この関数内で加工するため、コピーする
    cv_image = cv_image_original.copy()

    # 赤、白、黄色のボールの位置を取得
    red_balls = get_ball_position_by_color(cv_image, vertices, Color.RED)
    assert len(red_balls)>0, '赤玉が見つかりません'
    white_balls = get_ball_position_by_color(cv_image, vertices, Color.WHITE)
    assert len(white_balls)>0, '白玉が見つかりません'
    yellow_balls = get_ball_position_by_color(cv_image, vertices, Color.YELLOW)
    assert len(yellow_balls)>0, '黄玉が見つかりません'

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
    #print(rwy_balls)

    # 円の下の位置をボールの位置とするため、半径分下にずらす
    for ball in rwy_balls:
        ball[1] += ball[2]

    return rwy_balls
