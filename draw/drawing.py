import numpy as np
import cv2 as cv



drawing = False  # true if mouse is pressed
ix, iy = -1, -1


class Mouse():
    track_list_list = []
    track_list=[]
    # mouse callback function
    def __init__(self):
        self.img = np.zeros((255*3, 255*3, 3), np.uint8)
        for i in range(255*3):
            for j in range(255*3):
                self.img[i, j, 0] = np.uint8(255)
                self.img[i, j, 1] = np.uint8(255)
                self.img[i, j, 2] = np.uint8(255)
        self.track_list_list = []
        self.track_list=[]
    def draw_circle(self, event, x, y, flags, param):
        global ix, iy, drawing
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            self.track_list=[]
            slist=[x, y]
            self.track_list.append(slist)
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                slist = [x, y]
                self.track_list.append(slist)
                cv.circle(self.img, (x, y), 3, (0, 0, 0), -1)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            slist = [x, y]
            self.track_list.append(slist)
            self.track_list_list.append(self.track_list)
            cv.circle(self.img, (x, y), 3, (0, 0, 0), -1)



    def create_image(self):
        cv.namedWindow('press C continue')
        cv.setMouseCallback('press C continue', self.draw_circle)
        while (1):
            cv.imshow('press C continue', self.img)
            k = cv.waitKey(1) & 0xFF
            if k == ord('c'):
                # cv.destroyAllWindows()
                break


mn = Mouse()
mn.create_image()
