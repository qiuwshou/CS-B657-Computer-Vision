import cv2
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from math import *
import os
from os.path import join



class Obstacle_detctor:

    #initialize class variables
    #
    def __init__(self,dir1,dir2,f_size,t_ratio,t_scale,i_scale,flag):
        self.f_size = f_size
        self.t_ratio = t_ratio
        self.t_scale = t_scale
        self.i_scale = i_scale
        self.dir1 = dir1
        self.dir2 = dir2
        self.flag = flag


    #filter the surf features
    #return a list of filtered matches
    def filter_match(self,matches, base_kps, compare_kps,base_img,compare_img):
        good=[]
        row = base_img.shape[0]
        col = base_img.shape[1]
        row2 = compare_img.shape[0]
        col2 = compare_img.shape[1]
        for m in matches:
            base_x = base_kps[m.queryIdx].pt[0]
            base_y = base_kps[m.queryIdx].pt[1]
            compare_x = compare_kps[m.trainIdx].pt[0]
            compare_y = compare_kps[m.trainIdx].pt[1]
            size = base_kps[m.queryIdx].size
            size2 =compare_kps[m.trainIdx].size
            #select the features greater than a certain value f_size
            #select the features in the middle of the image
            #select the matches that the feature in the current image is lager than the one in the previous image
            if(base_x >= col/4 and base_x<= col*3/4 and base_y >= row/4 and base_y <= row*3/4 and size2>size and size>self.f_size ):
                if(compare_x >= col2/4 and compare_x<= col2*3/4 and compare_y >= row2/4 and compare_y <= row2*3/4):
                    good.append(m)
        return good

    #calculate the distance of template matching
    def cal_dist(self,img1, img2):
        ssd = 0
        for i in range(0,img1.shape[0]):
            for j in range(0,img1.shape[1]):
                ssd += pow(int(img1[i,j])-int(img2[i,j]),2)
        #normalize the ssd by size of template
        ssd = ssd/(img1.shape[0]*img1.shape[1])
        return ssd

    #extract the template from image by coordinates and image side
    def extract_temp(self,img,x,y,w):
        x_low = x-w
        x_up = x+w
        y_low = y-w
        y_up = y+w
        #only extract the templates that don't hit the image boundry, otherwise return a empty list
        if(x_low >= 0 and x_up < img.shape[1] and y_low >=0 and y_up < img.shape[0]):
            temp = img[y_low:y_up,x_low:x_up]
        else:
            temp = []
        return temp

    #filter the matches basing on the performance of template matching
    #return the good matches and the size of templates in previous and current images
    def temp_match(self,base_img,base_kps,compare_img,compare_kps,good):
        match =[]
        result = []
        size_record = []
        for m in good:
            base_pt = base_kps[m.queryIdx]
            compare_pt = compare_kps[m.trainIdx]
            #initialize the size of template in previous image basing on the size of surf feature
            size = floor(base_pt.size*self.i_scale/2)
            x = floor(base_pt.pt[0])
            y = floor(base_pt.pt[1])
            x2 = floor(compare_pt.pt[0])
            y2 = floor(compare_pt.pt[1])
            #extract templates from previous and current image with the same size
            temp1 = self.extract_temp(base_img,x,y,size)
            temp2_1 = self.extract_temp(compare_img,x2,y2,size)
            s_pair = [size,0]
            scale_size = 0
            if(len(temp1) > 0 and len(temp2_1) > 0):
                    scale_min = 9999
                    temp_min = 9999
                    temp_0 = self.cal_dist(temp1,temp2_1)
                    #scale the template from 1.1 to 1.5
                    for i in range(11,16):
                        scale = floor(i*size/10)
                        #extract a larger template by the scaled size in current image
                        temp2  = self.extract_temp(compare_img,x2,y2,scale)

                        if(len(temp2) > 0):
                            height, width = temp2.shape[:2]
                            #resize the template in the previous image by the scaled size
                            temp1_scale = cv2.resize(temp1,(width,height), interpolation = cv2.INTER_CUBIC)
                            temp_scale = self.cal_dist(temp1_scale,temp2)

                            #choose the scale that gives the best score for template matching
                            if(temp_scale<=temp_min):
                                temp_min = temp_scale
                                scale_min = i/10
                                scale_size = scale


                    #filter the features for following:
                    #if the scale is larger than a number(for example,1.2)
                    #and if the scaled templates improve the perforamnce of template match by a number(for example,0.8)
                    if(scale_min >= self.t_ratio and temp_min < self.t_scale*temp_0 and scale_min<2):
                        s_pair[1] = scale_size
                        size_record.append(s_pair)
                        match.append(m)
        result.append(match)
        result.append(size_record)
        return result


    #detect the feature of obstacle between a pair of previous and current images
    def obstacle_detect(self,img_base,img_compare):
        print("compare image1:%s "%img_base+"| image2:%s"%img_compare)
        base_img = cv2.imread(img_base,0)
        compare_img = cv2.imread(img_compare,0)
        #resize the two images
        base_img = cv2.resize(base_img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        compare_img = cv2.resize(compare_img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        #remove  gaussian noise
        #base_img = cv2.GaussianBlur(base_img,(5,5),5)
        #compare_img = cv2.GaussianBlur(compare_img,(5,5),5)

        base_surf = cv2.xfeatures2d.SURF_create()
        compare_surf = cv2.xfeatures2d.SURF_create()
        (base_kps, base_desc) = base_surf.detectAndCompute(base_img,None)
        (compare_kps, compare_desc) = compare_surf.detectAndCompute(compare_img,None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(base_desc,compare_desc)
        #select the features within some distance
        matches = [m for m in matches if m.distance <= 0.25]

        #use ransac
        #matches = self.RANSAC(base_desc,compare_desc,10)

        #output the results of feature filtering, template matching and template if flag = 1
        #return 1 if an obstacle is detected
        #return 0 if no obstacle is dectcted
        if(self.flag == 1):
            base_split = img_base.split("/")
            #generate filename
            #concatenate two images by features and matches
            name_base = base_split[-3]+"_"+base_split[-2]+"_"+base_split[-1][:-4]+"_"+img_compare[-12:-4]
            match_before = name_base + "_match_before.png"
            match_after = name_base + "_match_after.png"
            match_final = name_base + "_match_final.png"
            kps_before = name_base + "_kps_before.png"
            kps_after = name_base + "_kps_after.png"
            kps_final = name_base + "_kps_final.png"
            base_temp = name_base + "_previous_temp.png"
            compare_temp = name_base +"_previous_temp_scale.png"
            compare_temp_scaled = name_base + "_current_temp_scale.png"

            self.draw_kps(matches,base_img,compare_img,base_kps,compare_kps,kps_before)
            self.draw_match(matches,base_img,compare_img,base_kps,compare_kps,match_before)
            good_match = self.filter_match(matches, base_kps, compare_kps,base_img,compare_img)
            self.draw_kps(good_match,base_img,compare_img,base_kps,compare_kps,kps_after)
            self.draw_match(good_match,base_img,compare_img,base_kps,compare_kps,match_after)
            obstacle,window_size = self.temp_match(base_img,base_kps,compare_img,compare_kps,good_match)
            self.draw_kps(obstacle,base_img,compare_img,base_kps,compare_kps,kps_final)
            self.draw_match(obstacle,base_img,compare_img,base_kps,compare_kps,match_final)
            self.draw_temp(obstacle,base_img,compare_img,base_kps,compare_kps,window_size,base_temp,compare_temp,compare_temp_scaled)
        else:
            #filter the feature
            good_match = self.filter_match(matches, base_kps, compare_kps,base_img,compare_img)
            #do template matching
            obstacle,window_size = self.temp_match(base_img,base_kps,compare_img,compare_kps,good_match)
        if(len(obstacle)>0):
            return 1
        else:
            return 0
        return

    #draw the template of matches
    #draw the template of previous image
    #draw the resized template of previous image
    #draw teh scaled template of current image
    def draw_temp(self,matches,base_img,compare_img,base_kps,compare_kps,window_size,name1,name2,name3):
        if(len(matches)>0):
            for i in range(0,1):
                m = matches[i]
                pair = window_size[i]
                p_size = pair[0]
                s_size = pair[1]
                kp1 = base_kps[m.queryIdx]
                kp2 = compare_kps[m.trainIdx]
                x_low = ceil(kp1.pt[0]-p_size)
                x_up = ceil(kp1.pt[0]+p_size)
                y_low = ceil(kp1.pt[1]-p_size)
                y_up = ceil(kp1.pt[1]+p_size)
                temp_p = base_img[y_low:y_up,x_low:x_up]
                cv2.imwrite(name1,temp_p)

                x_low = ceil(kp2.pt[0]-s_size)
                x_up = ceil(kp2.pt[0]+s_size)
                y_low = ceil(kp2.pt[1]-s_size)
                y_up = ceil(kp2.pt[1]+s_size)
                temp_r = compare_img[y_low:y_up,x_low:x_up]
                cv2.imwrite(name3,temp_r)

                width = x_up - x_low
                height = y_up - y_low
                temp_c = cv2.resize(temp_p,(width,height), interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(name2,temp_c)
        return


    #draw the matching features
    def draw_kps(self,matches,base_img,compare_img,base_kps,compare_kps,name):
        b_kps = []
        c_kps = []
        if(len(matches) > 0):
            for m in matches:
                b_kps.append(base_kps[m.queryIdx])
                c_kps.append(compare_kps[m.trainIdx])
            b_feature = cv2.drawKeypoints(base_img,b_kps,None,(255,0,0),4)
            c_feature = cv2.drawKeypoints(compare_img,c_kps,None,(255,0,0),4)
            ##img1_kp = cv2.drawKeypoints(res1,kps1,None,(255,0,0),4)
            vis = np.concatenate((b_feature,c_feature),axis=1)
        else:
            # if there is no match, just concatenate two images
            vis = np.concatenate((base_img,compare_img),axis=1)
        cv2.imwrite(name, vis)
        return

    #draw a line between matching features
    def draw_match(self,matches,base_img,compare_img,base_kps,comapre_kps,name):
        if(len(matches)==0):
            vis = np.concatenate((base_img,compare_img),axis=1)
        elif(len(matches)>0):
            vis = cv2.drawMatches(base_img,base_kps,compare_img,comapre_kps,matches,None,(0,255,0),(0,0,255),flags = 2)
        cv2.imwrite(name,vis)
        return


    #calculate the distance between two decriptors
    def desc_dist(self,d1,d2):
        ssd = 0
        for i in range(0,len(d1)):
            #normalize the distance
            ssd += pow(d1[i]/255 - d2[i]/255,2)
        ssd = sqrt(ssd)
        return ssd

    #find the best match for each kp by ransac
    #we don't use for in this project because it's too slow
    def RANSAC(self,base_desc,compare_desc,iteration):
        min = [99999]*len(base_desc)
        train_id = [-1]*len(base_desc)
        for i in range(0,iteration):
            temp =[-1]*len(base_desc)
            for n in range(0,len(base_desc)):
                temp[n] = randint(0,len(compare_desc)-1)
            for n in range(0,len(base_desc)):
                dist = self.desc_dist(base_desc[n],compare_desc[temp[n]])
                #print(dist)
                if(dist < min[n] and dist < 0.0015):
                    min[n] = dist
                    train_id[n] = temp[n]
        record = self.create_match(train_id, min)
        return record

    #create a match basing by pairs of features
    def create_match(self,train_id,min):
        result = []
        for n in range(0,len(train_id)):
            m = cv2.DMatch()
            m.queryIdx = n
            m.trainIdx = train_id[n]
            m.imgIdx = -1
            m.distance = min[n]
            result.append(m)
        return result


    #generate all filenames under directories of obstacle data and non-obstacle data
    def get_file_name(self,dir1,dir2):
        result = []
        seq1 = [join(dir1,f) for f in os.listdir(dir1)]
        seq2 = [join(dir2,f) for f in os.listdir(dir2)]
        filename_no_obstacles = []
        filename_with_obstacles = []
        #10 sequences in non-obstacle set
        for i in range(0,len(seq1)):
            f=[join(seq1[i],f) for f in os.listdir(seq1[i])]
            filename_no_obstacles.append(f)
        #40 sequences in obstacle set
        for i in range(0,len(seq2)):
            f=[join(seq2[i],f) for f in os.listdir(seq2[i])]
            filename_with_obstacles.append(f)
        result.append(filename_no_obstacles)
        result.append(filename_with_obstacles)
        return result

    #detect obstacle in all the images
    #return false positive rate for non-obstacle data
    #return false negtive rate for obstacle data
    def compare_all(self):
        print("detecting the obstacle....")
        fn = 0
        fp = 0
        filename_no_obstacles, filename_with_obstacles  = self.get_file_name(self.dir1,self.dir2)
        count = 0

        for l in filename_with_obstacles:
            base_split = l[0].split("/")
            print("comparing images under "+base_split[4]+"/"+base_split[5])
            for i in range(0,len(l)-1):
                b=self.obstacle_detect(l[i],l[i+1])
                count += 1
                if(b==0):
                    fn += 1
        fn = fn/count

        count = 0
        for l in filename_no_obstacles:
            base_split = l[0].split("/")
            print("comparing images under "+base_split[4]+"/"+base_split[5])
            for i in range(0,len(l)-1):
                b=self.obstacle_detect(l[i],l[i+1])
                count += 1
                if(b == 1):
                    fp += 1
        fp = fp/count

        print("parameter:feature_size_lowbound:%d"%self.f_size+"|temp_ratio:%f"%self.t_ratio+"|scale_temp_match:%f"%self.t_scale+"|initial_temp_scale:%f"%self.i_scale)
        print("fp:%f"%fp+"|fn:%f"%fn)
        return





