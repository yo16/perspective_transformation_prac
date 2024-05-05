import numpy as np
import cv2

from constants import TABLE_LEN_LONG, TABLE_LEN_SHORT, CUSHION_WIDTH

# クッションを含む青い矩形４点から、
# 有効な位置へ変換するためのmatrixを返す
def get_matrix(points):
    # ４点
    # 左上、右上、右下、左下の順
    src_points = np.float32(points)
    #print(src_points)

    # ４点の変換後の位置
    # pointsはクッションの角なので、正確には変換後のクッションの角の位置
    dst_points = np.float32([
        [(-1)*CUSHION_WIDTH, TABLE_LEN_SHORT + CUSHION_WIDTH],
        [(-1)*CUSHION_WIDTH, (-1)*CUSHION_WIDTH],
        [TABLE_LEN_LONG + CUSHION_WIDTH, (-1)*CUSHION_WIDTH],
        [TABLE_LEN_LONG + CUSHION_WIDTH, TABLE_LEN_SHORT + CUSHION_WIDTH]
    ])

    # ホモグラフィ行列を計算する
    matrix = cv2.getPerspectiveTransform(src_points, np.array(dst_points))
    matrix = np.float32(matrix)
    #print(matrix)

    return matrix
