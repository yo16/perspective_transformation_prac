import numpy as np

# ボールの位置を、matrixで変換する
def transform_ball_positions(balls, matrix):
    # 元の点のリスト (x, y) 形式
    points = [(x, y) for x, y, _ in balls]

    # 同次座標に変換
    points_homogeneous = np.array([list(point) + [1] for point in points])

    # ホモグラフィ行列による変換
    transformed_points_homogeneous = np.dot(points_homogeneous, matrix.T)  # 行列を転置して乗算

    # 同次座標から通常の座標への変換
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, [2]]

    return transformed_points
