import cv2
cap = cv2.VideoCapture(0)


def display_corners(img,c1,c2,c3,c4):
    for i in range(c1,c2):
        for j in range(c3,c4):
            img[i][j] = [0,0,0]
    return img


def getSides():
    cv2.namedWindow('webcam',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('webcam', 800,800)
    imgs = []
    while True:
        ret,img = cap.read()
        imgs.append(img)

        # top left corner
        img = display_corners(img,200,210,352,412)
        img = display_corners(img,210,260,352,362)

        # bottom left corner
        img = display_corners(img,460,510,352,362)
        img = display_corners(img,510,520,352,412)

        # top right corner
        img = display_corners(img,200,210,622,672)
        img = display_corners(img,210,260,662,672)

        # bottom right corner
        img = display_corners(img,460,510,662,672)
        img = display_corners(img,510,520,622,672)

        cv2.imshow('webcam', img)
        k = cv2.waitKey(100)
        if k != -1:
            break
    return imgs
data = getSides()
def getData(color):
    f = open(color+"_rgb","a")
    for img in data[len(data)-1:]:
        for i in range(210,510):
            for j in range(362,662):
                f.write(str(img[i][j][2])+" "+str(img[i][j][1])+" "+str(img[i][j][0]))
                f.write("\n")
    f.close()
getData("white")
