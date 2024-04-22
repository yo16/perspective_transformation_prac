import cv2
import numpy as np


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
    return approx.tolist()


# perspective transformationで長方形へ変換
def cut_and_transform(original_image, points, width, height):
    src_points = np.float32(points)

    # 真上から見たビリヤード台の四隅の座標 (x, y) (通常は四角形)
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # ホモグラフィ行列を計算する
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 透視変換を実行する
    warped_image = cv2.warpPerspective(original_image, matrix, (width, height))

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
    width = 160*2
    height = 290*2

    original_image = cv2.imread(image_path)

    # 角を取得
    ret = get_corner_points(original_image)
    print(ret)

    # perspective transformationで長方形へ変換
    cut_and_transform(original_image, ret, width, height)



if __name__=="__main__":
    main()
