import numpy as np
import cv2
import pickle


# displays corners
def display_corners(img, c1, c2, c3, c4):
    for i in range(c1, c2):
        for j in range(c3, c4):
            img[i][j] = [0, 0, 0]
    return img


def side(img):

    # S
    for i in range(260, 270):
        for j in range(710, 800):
            img[i][j] = [0, 0, 0]
    for i in range(270, 355):
        for j in range(700, 710):
            img[i][j] = [0, 0, 0]
    for i in range(355, 365):
        for j in range(710, 790):
            img[i][j] = [0, 0, 0]
    for i in range(365, 450):
        for j in range(790, 800):
            img[i][j] = [0, 0, 0]
    for i in range(450, 460):
        for j in range(700, 790):
            img[i][j] = [0, 0, 0]

    # I
    for i in range(260, 270):
        for j in range(810, 910):
            img[i][j] = [0, 0, 0]
    for i in range(270, 450):
        for j in range(855, 865):
            img[i][j] = [0, 0, 0]
    for i in range(450, 460):
        for j in range(810, 910):
            img[i][j] = [0, 0, 0]

    # D
    for i in range(260, 460):
        for j in range(920, 930):
            img[i][j] = [0, 0, 0]
    for i in range(260, 270):
        for j in range(930, 970):
            img[i][j] = [0, 0, 0]
    for i in range(450, 460):
        for j in range(930, 970):
            img[i][j] = [0, 0, 0]
    for i in range(270, 280):
        for j in range(970, 1000):
            img[i][j] = [0, 0, 0]
    for i in range(440, 450):
        for j in range(970, 1000):
            img[i][j] = [0, 0, 0]
    for i in range(280, 300):
        for j in range(1000, 1010):
            img[i][j] = [0, 0, 0]
    for i in range(420, 440):
        for j in range(1000, 1010):
            img[i][j] = [0, 0, 0]
    for i in range(300, 420):
        for j in range(1010, 1020):
            img[i][j] = [0, 0, 0]

    # E
    for i in range(260, 270):
        for j in range(1030, 1130):
            img[i][j] = [0, 0, 0]
    for i in range(270, 355):
        for j in range(1030, 1040):
            img[i][j] = [0, 0, 0]
    for i in range(355, 365):
        for j in range(1030, 1130):
            img[i][j] = [0, 0, 0]
    for i in range(365, 450):
        for j in range(1030, 1040):
            img[i][j] = [0, 0, 0]
    for i in range(450, 460):
        for j in range(1030, 1130):
            img[i][j] = [0, 0, 0]

    return img


def side1(img):
    for i in range(260, 460):
        for j in range(1185, 1195):
            img[i][j] = [0, 255, 255]

    return img


def side2(img):
    for i in range(260, 270):
        for j in range(1140, 1240):
            img[i][j] = [0, 150, 255]
    for i in range(270, 355):
        for j in range(1230, 1240):
            img[i][j] = [0, 150, 255]
    for i in range(355, 365):
        for j in range(1140, 1240):
            img[i][j] = [0, 150, 255]
    for i in range(365, 450):
        for j in range(1140, 1150):
            img[i][j] = [0, 150, 255]
    for i in range(450, 460):
        for j in range(1140, 1240):
            img[i][j] = [0, 150, 255]

    return img


def side3(img):
    for i in range(260, 270):
        for j in range(1140, 1240):
            img[i][j] = [0, 50, 255]
    for i in range(270, 355):
        for j in range(1230, 1240):
            img[i][j] = [0, 50, 255]
    for i in range(355, 365):
        for j in range(1140, 1240):
            img[i][j] = [0, 50, 255]
    for i in range(365, 450):
        for j in range(1230, 1240):
            img[i][j] = [0, 50, 255]
    for i in range(450, 460):
        for j in range(1140, 1240):
            img[i][j] = [0, 50, 255]

    return img


def side4(img):
    for i in range(260, 460):
        for j in range(1230, 1240):
            img[i][j] = [0, 0, 255]
    for i in range(260, 370):
        for j in range(1140, 1150):
            img[i][j] = [0, 0, 255]
    for i in range(360, 370):
        for j in range(1150, 1230):
            img[i][j] = [0, 0, 255]

    return img


def side5(img):
    for i in range(260, 270):
        for j in range(1140, 1240):
            img[i][j] = [0, 0, 150]
    for i in range(270, 355):
        for j in range(1140, 1150):
            img[i][j] = [0, 0, 150]
    for i in range(355, 365):
        for j in range(1140, 1240):
            img[i][j] = [0, 0, 150]
    for i in range(365, 450):
        for j in range(1230, 1240):
            img[i][j] = [0, 0, 150]
    for i in range(450, 460):
        for j in range(1140, 1240):
            img[i][j] = [0, 0, 150]

    return img


def side6(img):
    for i in range(260, 270):
        for j in range(1140, 1240):
            img[i][j] = [0, 0, 75]
    for i in range(270, 450):
        for j in range(1140, 1150):
            img[i][j] = [0, 0, 75]
    for i in range(355, 365):
        for j in range(1140, 1240):
            img[i][j] = [0, 0, 75]
    for i in range(365, 450):
        for j in range(1230, 1240):
            img[i][j] = [0, 0, 75]
    for i in range(450, 460):
        for j in range(1140, 1240):
            img[i][j] = [0, 0, 75]

    return img


def get_sides(side_num, cap):
    cv2.namedWindow('webcam', cv2.WINDOW_NORMAL)
    imgs = []
    while True:
        ret, img = cap.read()
        imgs.append(img)

        img = side(img)
        if side_num == 1:
            img = side1(img)
        if side_num == 2:
            img = side2(img)
        if side_num == 3:
            img = side3(img)
        if side_num == 4:
            img = side4(img)
        if side_num == 5:
            img = side5(img)
        if side_num == 6:
            img = side6(img)

        # top left corner
        img = display_corners(img, 200, 210, 352, 412)
        img = display_corners(img, 210, 260, 352, 362)

        # bottom left corner
        img = display_corners(img, 460, 510, 352, 362)
        img = display_corners(img, 510, 520, 352, 412)

        # top right corner
        img = display_corners(img, 200, 210, 612, 672)
        img = display_corners(img, 210, 260, 662, 672)

        # bottom right corner
        img = display_corners(img, 460, 510, 662, 672)
        img = display_corners(img, 510, 520, 612, 672)

        cv2.imshow('webcam', img)
        k = cv2.waitKey(100)
        if k != -1:
            break
    return imgs


def predict_colors(blue, orange, yellow, white, red, green, c1, c2, c3, c4, image, forest):
    pr1 = 0
    pr2 = 0
    for i in range(c1+10, c2-10, 10):
        for j in range(c3+10, c4-10, 10):
            pixel = [image[i][j][2], image[i][j][1], image[i][j][0]]
            p = forest.predict([pixel])
            prob = forest.predict_proba([pixel])
            if p == 0:
                blue += 1
            elif p == 1:
                pr1 += prob[0][1]
                orange += 1
            elif p == 2:
                yellow += 1
            elif p == 3:
                white += 1
            elif p == 4:
                pr2 += prob[0][4]
                red += 1
            elif p == 5:
                green += 1
    cols = [blue, orange, yellow, white, red, green]
    max_col = 0
    for i in range(len(cols)):
        if cols[i] > cols[max_col]:
            max_col = i
    return max_col


def predict(data, side, forest, colors):
    blue, orange, yellow, white, red, green = 0, 0, 0, 0, 0, 0
    data[0] = predict_colors(blue, orange, yellow, white, red, green, 210, 310, 362, 462, side, forest)
    data[1] = predict_colors(blue, orange, yellow, white, red, green, 210, 310, 462, 562, side, forest)
    data[2] = predict_colors(blue, orange, yellow, white, red, green, 210, 310, 562, 662, side, forest)
    data[3] = predict_colors(blue, orange, yellow, white, red, green, 310, 410, 362, 462, side, forest)
    data[4] = predict_colors(blue, orange, yellow, white, red, green, 310, 410, 462, 562, side, forest)
    data[5] = predict_colors(blue, orange, yellow, white, red, green, 310, 410, 562, 662, side, forest)
    data[6] = predict_colors(blue, orange, yellow, white, red, green, 410, 510, 362, 462, side, forest)
    data[7] = predict_colors(blue, orange, yellow, white, red, green, 410, 510, 462, 562, side, forest)
    data[8] = predict_colors(blue, orange, yellow, white, red, green, 410, 510, 562, 662, side, forest)
    predictions = []
    for i in range(len(data)):
        predictions.append(colors[data[i]])
    return predictions


def main():

    colors = ["blue", "orange", "yellow", "white", "red", "green"]

    f = open("model", "rb")
    forest = pickle.load(f)
    f.close()

    cap = cv2.VideoCapture(0)

    s1imgs = get_sides(1, cap)
    s1imgs = s1imgs[-1]
    data = np.empty(shape=(9), dtype=int)
    print(predict(data, s1imgs, forest, colors))
    s2imgs = get_sides(2, cap)
    s2imgs = s2imgs[-1]
    data = np.empty(shape=(9), dtype=int)
    print(predict(data, s2imgs, forest, colors))
    s3imgs = get_sides(3, cap)
    s3imgs = s3imgs[-1]
    data = np.empty(shape=(9), dtype=int)
    print(predict(data, s3imgs, forest, colors))
    s4imgs = get_sides(4, cap)
    s4imgs = s4imgs[-1]
    data = np.empty(shape=9, dtype=int)
    print(predict(data, s4imgs, forest, colors))
    s5imgs = get_sides(5, cap)
    s5imgs = s5imgs[-1]
    data = np.empty(shape=9, dtype=int)
    print(predict(data, s5imgs, forest, colors))
    s6imgs = get_sides(6, cap)
    s6imgs = s6imgs[-1]
    data = np.empty(shape=9, dtype=int)
    print(predict(data, s6imgs, forest, colors))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
