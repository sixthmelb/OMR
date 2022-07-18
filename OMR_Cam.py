import cv2
import numpy as np
import utlis


###############################
#path = input("Nama File LJK : ")
path = "123.jpeg"
#frm = input("Format File [jpg / jpeg] : ")
wImg = 700
hImg = 700
questions = 5
choices = 5
ans = [1,2,0,1,4]

webcamFeed = True
cameraNo = 0
###############################


cap = cv2.VideoCapture(cameraNo)
cap.set(70,200)



while True:
    if webcamFeed: success, img = cap.read()
    else: 
        #img = cv2.imread(f"{path}.{frm}") #fungsi file drive
        img = cv2.imread(path) #fungsi kamera



    #Preprocessing 
    img = cv2.resize(img, (wImg,hImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)


    try:
        #Mencari Countour Gambar
        contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

        #Mencari Area Kotak
        rectCon = utlis.rectCountour(contours)
        biggestContour = utlis.getCornerPoints(rectCon[0])
        gradePoints = utlis.getCornerPoints(rectCon[1])
        #print(biggestContour)

        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0,255,0), 20)
            cv2.drawContours(imgBiggestContours, gradePoints, -1, (255,0,0), 20)

            biggestContour = utlis.reorder(biggestContour)
            gradePoints = utlis.reorder(gradePoints)

            #Fungsi Get Kotak Jawaban
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0],[wImg,0],[0,hImg],[wImg,hImg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored = cv2.warpPerspective(img,matrix,(wImg,hImg))

            #Fungsi Get Kotak Nilai
            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
            matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
            imgGradeDisplay = cv2.warpPerspective(img,matrixG,(325,150))
            #cv2.imshow("Grade", imgGradeDisplay)

            #Treshold
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgTresh = cv2.threshold(imgWarpGray,175,255,cv2.THRESH_BINARY_INV)[1]


            #Split Kotak
            boxes = utlis.splitBoxes(imgTresh)
            #cv2.imshow("Test", boxes[4])
            #print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))


            #logic of pixel :')
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0

            for image in boxes:
                totalPixel = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixel
                countC += 1
                if (countC == choices) : countR += 1 ;countC=0
            #print(myPixelVal)

            #logic dept math marking
            myIndex = []
            for x in range (0, questions) :
                arr = myPixelVal[x]
                #print("Array", arr)
                myIndexVal = np.where(arr==np.amax(arr))
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            #print(myIndex)


            #logic for grading
            grading =[]
            for x in range (0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            #print(grading)

            score = (sum(grading)/questions) *100 #final grade
            #print(score)

            #Logic Display Answer On P.T
            imgResult = imgWarpColored.copy()
            imgResult = utlis.showAnswers(imgWarpColored,myIndex,grading,ans,questions,choices)
            #overlay
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = utlis.showAnswers(imgRawDrawing,myIndex,grading,ans,questions,choices)
            #inverse Display with overlay
            invMatrix = cv2.getPerspectiveTransform(pt2,pt1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawing,invMatrix,(wImg,hImg))
            #final image
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score)),(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
            invMatrixG = cv2.getPerspectiveTransform(ptG2,ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade,invMatrixG,(wImg,hImg))
            #cv2.imshow("Grade", imgRawGrade)
            imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)

        imgBlank = np.zeros_like(img)
        imageArray = ([img,imgGray,imgBlur,imgCanny],
        [imgContours,imgBiggestContours,imgWarpColored,imgTresh],
        [imgResult,imgRawDrawing,imgInvWarp,imgFinal])

    except:

        imgBlank = np.zeros_like(img)
        imageArray = ([imgBlank,imgBlank,imgBlank,imgBlank],
        [imgBlank,imgBlank,imgBlank,imgBlank],
        [imgBlank,imgBlank,imgBlank,imgBlank])


    #labels = [["Original","Gray","Blur","Canny"],
    #["Contours", "Biggest Con", "Warp", "Treshold"],
    #["Result","Raw Drawing", "Inverse Warp", "Final Image"]]
    imgStacked = utlis.stackImages(imageArray,0.3)



    cv2.imshow(f"Hasil Periksa {path}", imgFinal)
    cv2.imshow(f"File {path}", img)
    #cv2.imshow("Stacked Images", imgStacked)
    #Fungsi Full File Drive
    #key = cv2.waitKey(0)
#
    #if key == 27: #if you press ESC you die
    #    break
    #    cv2.destroyAllWindows()
    #    #cv2.waitKey(300)
    #elif key == ord('s'):
    #    cv2.imwrite((f"Final-{path}.jpg"), imgFinal)
    #    cv2.destroyAllWindows()



    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("FinalResult.jpg", imgFinal)
        cv2.waitKey(300)



