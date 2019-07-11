import cv2
import numpy as np

#该类中的方法是利用surf进行图像的配准对齐工作
class surf_(object):
    def __init__(self,img1,img2):
        self.img1 = cv2.imread(img1)
        self.img2 = cv2.imread(img2)

    def surf_kp(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        suft = cv2.xfeatures2d_SURF.create()
        kp, des = suft.detectAndCompute(image, None)
        kp_image = cv2.drawKeypoints(gray_image, kp, None)
        return kp_image, kp, des

    def get_good_match(self,des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def suftImageAlignment(self):
        _, kp1, des1 = self.surf_kp(self.img1)
        _, kp2, des2 = self.surf_kp(self.img2)
        goodMatch = self.get_good_match(des1, des2)
        if len(goodMatch) > 4:
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 4
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
            imgOut = cv2.warpPerspective(self.img2, H, (self.img1.shape[1],self.img1.shape[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return imgOut, H, status
