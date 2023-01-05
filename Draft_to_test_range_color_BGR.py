import cv2 as cv
import numpy as np

# img = cv.imread('1.jpg')
# img = cv.GaussianBlur(img , (5,5),0)
# img = cv.resize(img, (700, 700))
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cap = cv.VideoCapture(1)
if not cap.isOpened():
    print('Can not open camera')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    tag = 0
    
    # img = cv.resize(img, (700, 500))
   

    # img = cv.GaussianBlur(img, (3,3), 0)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # img_c = img.copy()
    # img_c = cv.GaussianBlur(img_c, (3,3), 0)
    # gray = cv.cvtColor(img_c,cv.COLOR_BGR2GRAY)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    # gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    # gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)

    # mask = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)
    # # # # white 
    # high = np.array([180, 100, 255])
    # low = np.array([0, 0, 160])
    # # orange
    # low = np.array([0, 50, 180])
    # high = np.array([15, 255, 255])
    # # green
    # low = np.array([40, 90, 100])
    # high = np.array([80, 255, 255])
    # blue
    # low = np.array([100, 90, 80])
    # high = np.array([140, 255, 255])

    # red
    # low1 = np.array([0, 100, 90])
    # high1 = np.array([15, 250, 200])
    # low2 = np.array([170, 100, 90])
    # high2 = np.array([180, 250, 200])
    # red1 = cv.inRange(hsv, low1, high1)
    # red2 = cv.inRange(hsv, low2, high2)
    # mask = red1 + red2

    
    # # yellow
    # low = np.array([20, 100, 100])
    # high = np.array([40, 255, 255])
    # colors_hsv = [('white', [0, 0, 160], [180, 80, 255]), ('orange',[0, 60, 200], [10, 255, 255]), ('green', [40, 90, 100], [90, 255, 255]),
    #  ('blue', [100, 90, 80], [140, 255, 255]), ('yellow', [20, 50, 100], [50, 250, 250]), ('red', [[0, 90, 110], [7, 220, 220]], [[170, 80, 100], [180, 220, 220]])]
    colors_hsv = [('white', [0, 0, 150], [180, 100, 255]), ('orange',[0, 60, 180], [15, 255, 255]),('green', [40, 90, 100], [90, 255, 255]),('blue', [100, 90, 80], [140, 255, 255]),('yellow', [20, 50, 90], [50, 255, 255]),
    ('red', [[0, 90, 110], [15, 250, 220]], [[170, 90, 100], [180, 250, 220]])]

    blob_colors = []
    face = np.array([0,0,0,0,0,0,0,0,0])
    for i, color in enumerate(colors_hsv):
        if color[0] != 'red':
            low = np.array(color[1])
            high = np.array(color[2])
            mask = cv.inRange(hsv, low, high)
        
        
        else:
            low1 = np.array(color[1][0])
            high1 = np.array(color[1][1])
            low2 = np.array(color[2][0])
            high2 = np.array(color[2][1])
            red1 = cv.inRange(hsv, low1, high1)
            red2 = cv.inRange(hsv, low2, high2)
            mask = red1 + red2

        kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        # mask = cv.inRange(hsv, low, high)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel,iterations=1)

        contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[0:20]


        for contour in contours:
            area = cv.contourArea(contour)
            
            if 1000 < area < 3000:
                perimeter = cv.arcLength(contour, True)
                epsilon = 0.09 * perimeter
                approx = cv.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    x,y,w,h = cv.boundingRect(contour)
                    cv.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2, 2)
                    color = np.array(cv.mean(img[y:y+h,x:x+w])).astype(int)
                    print(color)
                    blob_colors.append(color)
    for i in range(len(blob_colors)):
            if blob_colors[i][0] > 180 and blob_colors[i][1] > 180 and blob_colors[i][2] > 150:
                blob_colors[i][3] = 1
            if blob_colors[i][0] < 120 and blob_colors[i][1] < 120 and blob_colors[i][2] > 150:
                blob_colors[i][3] = 2
            if blob_colors[i][0] < 120 and blob_colors[i][1] > 180 and blob_colors[i][2] < 100:
                blob_colors[i][3] = 3
            if blob_colors[i][0] > 100 and blob_colors[i][1] > 50 and blob_colors[i][2] < 50:
                blob_colors[i][3] = 4
            if blob_colors[i][0] > 30 and blob_colors[i][1] > 180 and blob_colors[i][2] > 150 and np.abs(blob_colors[i][0]-blob_colors[i][2]) > 100:
                blob_colors[i][3] = 5
            if blob_colors[i][0] < 100 and blob_colors[i][1] < 100 and blob_colors[i][2] > 100 and blob_colors[i][0]>blob_colors[i][1] :
                blob_colors[i][3] = 6
    print(blob_colors)
    cv.imshow("img", img)
    # cv.imshow("img", mask)


    if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty('img', cv.WND_PROP_VISIBLE) < 1: 
            break



cap.release()
cv.destroyAllWindows()
# #     if len(blob_colors) == 9:
# #         #print(blob_colors)
# #         for i in range(9):
# #             #print(blob_colors[i])
# #             if blob_colors[i][0] > 180 and blob_colors[i][1] > 180 and blob_colors[i][2] < 180:
# #                 blob_colors[i][3] = 1
# #                 face[i] = 1
# #                 count1 +=1
# #                 print('count1',count1)
# #                 print(blob_colors[i])
# #             elif blob_colors[i][0] < 150 and blob_colors[i][1] < 150 and blob_colors[i][2] > 180 and blob_colors[i][2] > blob_colors[i][0] and blob_colors[i][2] > blob_colors[i][1]:
                
# #                 blob_colors[i][3] = 2
# #                 face[i] = 2
# #                 count2 +=1
# #                 print('count2',count2)
# #                 print(blob_colors[i])
# #             elif blob_colors[i][0] <130 and 120 < blob_colors[i][1] < 170 and blob_colors[i][2] < 100 and np.abs(blob_colors[i][0] - blob_colors[i][2]) < 80:
# #                 blob_colors[i][3] = 3
                
# #                 face[i] = 3
# #                 count3 +=1
# #                 print('count3',count3)
# #                 print(blob_colors[i])
# #             elif blob_colors[i][0] < 150 and blob_colors[i][1] > 150 and 150 < blob_colors[i][2] < 190 and blob_colors[i][2] > blob_colors[i][0]:
# #                 blob_colors[i][3] = 4
# #                 face[i] = 4
# #                 count4 +=1
# #                 print('count4',count4)
# #                 print(blob_colors[i])
# #             elif blob_colors[i][0] < 150 and blob_colors[i][1] < 120 and  blob_colors[i][2] < 70 and blob_colors[i][1] > blob_colors[i][2]:
# #                 blob_colors[i][3] = 5
# #                 face[i] = 5
# #                 count5 +=1
# #                 print('count5',count5)
# #                 print(blob_colors[i])
# #             elif blob_colors[i][0] < 150 and blob_colors[i][1] > 150 and 150 < blob_colors[i][2] < 190 and blob_colors[i][2] > blob_colors[i][0]:
# #                 blob_colors[i][3] = 4
# #                 face[i] = 4
# #             elif blob_colors[i][0] < 150 and blob_colors[i][1] < 120 and  blob_colors[i][2] < 70 and blob_colors[i][1] > blob_colors[i][2]:
# #                 blob_colors[i][3] = 5
# #                 face[i] = 5
# #             elif blob_colors[i][0] < 120 and blob_colors[i][1] < 120 and  blob_colors[i][2] > 80 and np.abs(blob_colors[i][0] - blob_colors[i][1]) < 30:
# #                 blob_colors[i][3] = 6
# #                 face[i] = 6
# #                 count6 +=1
# #                 print('count6',count6)
# #                 print(blob_colors[i])

                    


# #     cv.imshow('img', img)
# #     # cv.imshow('img',mask)
# #     if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty('img', cv.WND_PROP_VISIBLE) < 1: 
# #         break

# # cap.release()
# # cv.destroyAllWindows()


# # # import numpy as np
# # # import cv2

# # # img = cv2.imread('2.jpg')
# # # img = cv2.resize(img, (1500, 1000))
# # # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # # a = img[390:540, 450:630]
# # # print(np.array(cv2.mean(a)).astype(int))
# # # # print(img.shape)
# # # cv2.imshow('img', a)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()

# # # a = np.array([[[1,9], [4, 5], [6, 7]], [[3, 7], [6,1], [8,5]]])

# # # c = np.array([[1, 8, 5, 7], [9, 6, 7,1], [5,1,6,3]])

# # # c = c[c[:, 1].argsort()]
# # # print(c)

# importing libraries
