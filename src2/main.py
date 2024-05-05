# 画像から、玉の座標を返す
# 横方向で、テーブル(球が動く有効なエリア)の左上の位置を(0,0)とする座標系。単位はmm
import os
import sys
import cv2
import numpy as np
import json

from get_corner_points import get_corner_points 
from get_ball_positions import get_ball_positions
from get_matrix import get_matrix
from transform_ball_positions import transform_ball_positions

def main(image_path):
    # 画像を読み込む
    original_image = cv2.imread(image_path)

    # ビリヤード台の角を取得
    # この角は、クッションの角なので、テーブルサイズより少し広い
    corners = get_corner_points(original_image)
    #print(corners)

    # 角の位置を元に、座標変換用の行列を取得
    matrix = get_matrix(corners)

    # ボールの位置を取得
    rwy_balls = get_ball_positions(original_image, corners)
    #print(rwy_balls)

    # 行列を適用させて、変換後座標を得る
    rwy_ball_positions = transform_ball_positions(rwy_balls, matrix)

    return rwy_ball_positions


if __name__=='__main__':
    image_path = './images/sample1.jpg'

    # コマンドライン引数を受け取る
    args = sys.argv
    if (len(args) > 1):
        image_path = args[1]
    #print(image_path)

    # ボールの位置を取得
    rwy_ball_pos = main(image_path)


    # 出力ファイル
    out_path = os.path.join(
        os.path.dirname(image_path),
        os.path.splitext(
            os.path.basename(image_path)
        )[0] + "_balls.json"
    )
    # 出力
    dict_out = {
        "red": {"x": rwy_ball_pos[0][0], "y": rwy_ball_pos[0][1]},
        "white": {"x": rwy_ball_pos[1][0], "y": rwy_ball_pos[1][1]},
        "yellow": {"x": rwy_ball_pos[2][0], "y": rwy_ball_pos[2][1]}
    }
    with open(out_path, "w") as f:
        json.dump(dict_out, f, ensure_ascii=False, indent=4)
