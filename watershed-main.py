#http://www.pyimagesearch.com/2015/11/02/watershed-opencv/
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
import os,sys
import time
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
        #help="path to input image")
#args = vars(ap.parse_args())
 
# load the image and perform pyramid mean shift filtering
# to aid the thresholding step

input_path = os.getcwd() + "/rice_data/"
#input_path = os.getcwd() + "/"
output_path = os.getcwd() + "/Output/watershed-part/"
#output_path = os.getcwd() + "/"
#method
TF_show = True
TF_write = False
TF_global_equalization = False
TF_median_filter = False
TF_remove_counted = False
TF_ignore_background = True
TF_find_average_r = True
TF_count_edge = False

only_show = ('False' , 'IMAG1565[275].jpg')[1]

#parameter
n_part = 4
circle_color = (0, 255, 0)
GREEN = (0,255,0)
BLUE = (255,0,0)
RED = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)
median_f = 25

switch_color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,125,125),(125,255,125),(125,125,255),(255,255,125),(125,255,255),(255,125,255),(125,0,125),(0,125,125),(125,125,0)]


PyrMeanShift_time = 0
Color2gray_time = 0
Localmaxmin_time = 0
Threshold_time = 0
Mix_time = 0
Draw_circle_time = 0



for i,fileName in enumerate(os.listdir(input_path)):
      
    dele = []
    circle = []
    circle_r = (0,0)
    rice_number = 0
    
    if(only_show!='False'):
	if i > 0:
	    break
	fileName = only_show    
    
    if(fileName[-3:]!='jpg' and fileName[-3:]!='JPG' and fileName[-3:]!='jpeg'):
	print "Wrong format file: "+fileName
	continue    
    
    start_time = time.time()
    r_list_local = []
    r_list = []
    position_list = []
    edge_number = []
    
    print fileName
  
    #image_ori = cv2.resize(cv2.imread(input_path+fileName), (0,0), fx=0.3, fy=0.3)
    
    image_ori = cv2.imread(input_path+fileName)
    image_color_part = image_ori.copy()
    
    if(TF_global_equalization):
	image_ori = global_equalization(image_ori)
      
   
    t2_img = image_ori.copy()

    tmp_img = image_ori.copy()
    
    if(TF_median_filter):
	tmp_img = cv2.medianBlur(tmp_img,median_f)
    
    image_gray = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
    
    
   
    
    #find max and min value of gray image
    max_value = np.amax(image_gray)
    min_value = np.amin(image_gray)
    
    height_ori,width_ori,channel_ori = image_ori.shape

    #contour_image = np.zeros((height_ori,width_ori), np.uint8)	        
    
    shifted = np.zeros((height_ori,width_ori,3), np.uint8)
    thresh = np.zeros((height_ori,width_ori), np.uint8)
    #print type(gray),type(gray[1*(height_ori/n_part) : (1+1)*(height_ori/n_part) ,  1*(width_ori/n_part) : (1+1)*(width_ori/n_part)])
    #sys.exit(0)
    
    #sys.exit(0)
    
    windowSize_v = int(height_ori/n_part)
    windowSize_h = int(width_ori/n_part)
    print 'window size : %d * %d' % (windowSize_v,windowSize_h)    
    
    N = 0
    co = 0
    for part_v in range(n_part*2):
	part_v = part_v/2 + part_v%2*0.5
	if(part_v > n_part-1):
	    break
	for part_h in range(n_part*2):
	    part_h = part_h/2 + part_h%2*0.5
	    if(part_h > n_part-1):
		break
	    
	    #print 'row : %.1f    column : %.1f' % (part_v,part_h)
    
    #for pv in range(0,height_ori*(n_part-1)/n_part,148):
	#for ph in range(0,width_ori*(n_part-1)/n_part,263):
    
	    shift_v = int(part_v*windowSize_v)
	    shift_h = int(part_h*windowSize_h)
	    
	    
	    
	    image = tmp_img[shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h]
	    #image = tmp_img[ pv : pv+height_ori/n_part , ph : ph+width_ori/n_part ]
	   
	    height,width,channel = image.shape
	    	    
	    shifted[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ] = cv2.pyrMeanShiftFiltering(image, 21, 51)	    
	    #shifted[ pv : pv+height_ori/n_part , ph : ph+width_ori/n_part ] = cv2.pyrMeanShiftFiltering(image, 21, 51)
	   
	   
	    # convert the mean shift image to grayscale, then apply
	    # Otsu's thresholding
	   
	    gray = cv2.cvtColor(shifted[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ], cv2.COLOR_BGR2GRAY)
	    #gray = cv2.cvtColor(shifted[ pv : pv+height_ori/n_part , ph : ph+width_ori/n_part ], cv2.COLOR_BGR2GRAY)
	   	
	    	 
	    local_max = np.amax(gray)
	    local_min = np.amin(gray)
	 
	    if(TF_ignore_background):
		if((local_max - local_min) < (max_value - min_value)/2):
		    #cv2.putText(image_ori, "[Background]", (int(part_h*(width_ori/n_part)+15), int(part_v*(height_ori/n_part) + (height_ori/n_part)/2)) ,cv2.FONT_HERSHEY_SIMPLEX, 1.9, (255, 0, 255), 3)
		    #cv2.putText(image_ori, str(local_max - local_min) +' @@ ' +str(max_value - min_value), (int(part_h*(width_ori/n_part)+15), int(part_v*(height_ori/n_part) + (height_ori/n_part)/2)) ,cv2.FONT_HERSHEY_SIMPLEX, 1.9, (255, 0, 255), 3)			
		    continue
		#else:
		    #cv2.putText(image_ori, str(local_max - local_min) +' - ' +str(max_value - min_value), (int(part_h*(width_ori/n_part)+15), int(part_v*(height_ori/n_part) + (height_ori/n_part)/2)) ,cv2.FONT_HERSHEY_SIMPLEX, 1.9, (255, 0, 255), 3)
	
	    
	    
	    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	    
	    thresh[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ] = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]
	    #thresh[pv : pv+height_ori/n_part , ph : ph+width_ori/n_part] = cv2.threshold(gray, 0, 255,
	            #cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	    
	    
	    #tmp = thresh.copy()
	    tmp = thresh[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ].copy()
	    #tmp = thresh[pv : pv+height_ori/n_part , ph : ph+width_ori/n_part].copy()		    
	    
	    # compute the exact Euclidean distance from every binary
	    # pixel to the nearest zero pixel, then find peaks in this
	    # distance map
	    
	    D = ndimage.distance_transform_edt(tmp)
	    localMax = peak_local_max(D, indices=False, min_distance=20,labels=tmp)	    
	    	    
	    # perform a connected component analysis on the local peaks,
	    # using 8-connectivity, then appy the Watershed algorithm
	    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	    labels = watershed(-D, markers, mask=tmp)
	    #print("[INFO] {} unique segments found".format(rice_number))
	    
	    		       
	    # loop over the unique labels returned by the Watershed
	    # algorithm
	    
	    top = shift_v 
	    bottom = shift_v + windowSize_v
	    left = shift_h 
	    right = shift_h + windowSize_h	    
	    
	    #draw_circle_time = time.time()
	    for label in np.unique(labels):
		
		    
		    t_img = t2_img.copy()
		    # if the label is zero, we are examining the 'background'
		    # so simply ignore it
		    if label == 0:
			    continue
	     
		    # otherwise, allocate memory for the label region and draw
		    # it on the mask
		    mask = np.zeros(gray.shape, dtype="uint8")
		    mask[labels == label] = 255
		    #print mask
		    #cv2.imshow('mask', mask)
		    #cv2.waitKey(0)	
		    
		    # detect contours in the mask and grab the largest one
		    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		    #print cnts[0][0][0][0],cnts[0][0][0][1]
		    #cv2.imshow('Binary:'+str(N), cv2.resize(thresh, (0,0), fx=0.3, fy=0.3))
		    #cv2.waitKey(0)
		    
		    c = max(cnts, key=cv2.contourArea)
		   
	     
		    # draw a circle enclosing the object
		    ((x, y), r) = cv2.minEnclosingCircle(c)
		    x = int(x)
		    y = int(y)
		    r = int(r)
		    
		        
		    circle_color = (0, 255, 0)
		    over_cover = 0
		    #k_img = t_img.copy()
		    
		   
				
		    
		    
		    #top = pv
		    #bottom = pv+(height_ori/n_part) 
		    #left = ph
		    #right = ph+(width_ori/n_part)		    
		    
		    #if( False or not ( ((int(x+ph)-r)<left and left!=0) or ((int(x+ph)+r)>right and right!=width_ori) or ((int(y+pv)-r)<top and top!=0) or ((int(y+pv)+r)>bottom and bottom!=height_ori) ) ):
			#cv2.circle(image_ori, (int(x+ph), int(y+pv)), int(r), circle_color, 2)
		    #else:
			#r_list_local.remove((int(x+ph),int(y+pv),int(r)))
			#N-=1		    
		    
		    #ignore the circle which is too small or out of inner boundary (not image's boundary) 
		    if( r > 10 and not ( ((x+shift_h-r)<left and left!=0) or ((x+shift_h+r)>right and right!=width_ori) or ((y+shift_v-r)<top and top!=0) or ((y+shift_v+r)>bottom and bottom!=height_ori) ) ):
		    #if( r > 10 ):
			
			if(TF_count_edge):
			    peri = cv2.arcLength(c, True)
			    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
					
			if(True and not TF_find_average_r):
			    if(r_list == []):
				if(TF_count_edge):
				    edge_number.append(len(approx))	
				r_list.append((x+shift_h,y+shift_v,r))
			    else:
				for other_r in  r_list:
				    if( pow(other_r[0]-int(x+shift_h),2) + pow(other_r[1]-int(y+shift_v),2) < pow(min(other_r[2],r),2) ):
				    #if(other_r[0] == int(x+shift_h) and other_r[1] == int(y+shift_v)):
					over_cover = 1
					break
				if(not over_cover):
				    if(TF_count_edge):
					edge_number.append(len(approx))	
				    r_list.append((x+shift_h,y+shift_v,r))
		    
			    if(over_cover):
				continue		      
			#print x,y,r,int((part_h+1)*(width_ori/n_part)),int((part_v+1)*(height_ori/n_part)),thresh[int(y+part_v*(height_ori/n_part)),int(x+part_h*(width_ori/n_part))]
			#print int(y+part_v*(height_ori/n_part))-int(r),int(y+part_v*(height_ori/n_part))+int(r),int(x+part_h*(width_ori/n_part))-int(r) , int(x+part_h*(width_ori/n_part))+int(r)
			
			if( position_list.count( (x+shift_h,y+shift_v) ) == 0 ):
			    position_list.append( (x+shift_h,y+shift_v) )
			    r_list_local.append( [ r , (x+shift_h,y+shift_v) , cnts , (shift_v , shift_v + windowSize_v ,  shift_h , shift_h + windowSize_h),GREEN,switch_color[co%15] ] )
			    co+=1
			
			#circle.append( ( r , (x+shift_h, y+shift_v) , cnts , (shift_v , shift_v + windowSize_v ,  shift_h , shift_h + windowSize_h) ) )
			#circle_r = (circle_r[0]+int(r),circle_r[1]+1)
			#print circle_r
			
			
			if(TF_count_edge):
			    if(len(approx)>8):
				cv2.circle(image_ori, (x+shift_h, y+shift_v), r, RED, 2)
			    else:
				cv2.circle(image_ori, (x+shift_h, y+shift_v), r, GREEN, 2)
			else:
			    if(not TF_find_average_r):	
				cv2.circle(image_ori, (x+shift_h, y+shift_v), r, GREEN, 2)
			
			#cv2.imshow('Ori:'+str(N), cv2.resize(image_ori, (0,0), fx=0.3, fy=0.3))
			#cv2.imshow('Bin:'+str(N), cv2.resize(thresh, (0,0), fx=0.3, fy=0.3))
			#cv2.waitKey(0)
			
			
			
			if(TF_remove_counted):
			    mask_x, mask_y = np.where(mask!=0)
			    pts = zip(mask_x, mask_y)			
			    dele.append((pts,shift_v,shift_h))			    
			    for d in dele:
				for xy in d[0]:	    			    
				    thresh[xy[0]+d[1],xy[1]+d[2]] = 0
				
			#cv2.imshow('Bin:'+str(N), cv2.resize(thresh, (0,0), fx=0.3, fy=0.3))
			#cv2.waitKey(0)						
		    #else:
			#r_list_local.remove((x+shift_h,y+shift_v,r))
			#cv2.circle(t2_img, (int(x+part_h*(width_ori/n_part)), int(y+part_v*(height_ori/n_part))), int(r), (255,255,0), 2)
			
		    #cv2.imshow('Wstershed:'+str(N), cv2.resize(image_ori, (0,0), fx=0.3, fy=0.3))
		    #cv2.imshow('Binary:'+str(N), cv2.resize(thresh, (0,0), fx=0.3, fy=0.3))
		    #cv2.waitKey(0)    		
		    
	    #Draw_circle_time += time.time() - draw_circle_time	  	    
			
    #cv2.imshow('Ori', cv2.resize(cv2.imread(os.getcwd()+'/rice_data-1/'+fileName), (0,0), fx=0.3, fy=0.3))
    
    if(TF_count_edge):
	edge_number.sort()
	edge_list = []
	n=0
	t_e=edge_number[0]	
	for e in edge_number:
	    if(e==t_e):
		n+=1
	    else:
		edge_list.append((t_e,n))
		n=1
		t_e = e
	edge_list = sorted(edge_list,reverse=True,key=lambda x:x[1])
	print edge_list
	
	
 #[ r , (x+shift_h,y+shift_v) , cnts , (shift_v , shift_v + windowSize_v ,  shift_h , shift_h + windowSize_h),GREEN ]	
	
    if(TF_find_average_r):
	#r = circle_r[0]/circle_r[1]
	
	#for c in circle:
	    #if(c[1] < 1.2*r and c[1] > 0.8*r):
		#cv2.circle(image_ori, c[0], c[1], circle_color, 2)
	    #else:
		#N-=1
	
	r_list_local.sort(reverse=True)
	
	median_index = (len(r_list_local)+1)/2 
	mean_r = r_list_local[median_index][0]
	cover_list = []
	print 'mean_r : ' + str(mean_r)
	
	start_draw = False
	co=1
	for each_circle in r_list_local:
	    if( not start_draw and each_circle[0] < 1.2*mean_r ):
		start_draw = True
		print 'start_draw = True'
	    if( not start_draw):
		cv2.circle(image_ori, each_circle[1], each_circle[0],  BLUE, 2)
		blank_image = np.zeros((windowSize_v,windowSize_h,3), np.uint8)
		blank_image[:,:] = BLACK
		cv2.drawContours(blank_image,each_circle[2],0, WHITE ,-1)
		#print type(blank_image),type(image_ori)
		#cv2.imshow('blank_image', blank_image)
		#cv2.waitKey(0)
		tmp_gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
		tmp_thresh = cv2.threshold(tmp_gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		#cv2.imshow('thre_image', tmp_thresh)
		#cv2.imshow('gray_image', tmp_gray)
		#cv2.waitKey(0)			
		tmp_D = ndimage.distance_transform_edt(tmp_thresh)
		tmp_localMax = peak_local_max(tmp_D, indices=False, min_distance=20,labels=tmp_thresh)			    
		tmp_markers = ndimage.label(tmp_localMax, structure=np.ones((3, 3)))[0]
		tmp_labels = watershed(-tmp_D, tmp_markers, mask=tmp_thresh)
		del tmp_D,tmp_localMax,tmp_markers	
		#print 'big circle'
		for tmp_label in np.unique(tmp_labels):
		    tmp_mask = np.zeros(tmp_gray.shape, dtype="uint8")
		    tmp_mask[tmp_labels == tmp_label] = 255
		    tmp_cnts = cv2.findContours(tmp_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		    tmp_c = max(tmp_cnts, key=cv2.contourArea)
		    ((tmp_x, tmp_y), tmp_r) = cv2.minEnclosingCircle(tmp_c)
		    tmp_x = int(tmp_x)
		    tmp_y = int(tmp_y)
		    tmp_r = int(tmp_r)
		    if(tmp_r>each_circle[0]):
			#print 'tmp_r:'+str(tmp_r)
			continue
		    r_list_local.append( [ tmp_r , (tmp_x+each_circle[3][2],tmp_y+each_circle[3][0]) , tmp_cnts ,( each_circle[3][0] , each_circle[3][1] , each_circle[3][2] , each_circle[3][3] ),RED,switch_color[co%15] ] )
		    co+=1
		    r_list_local[median_index+1:] = sorted(r_list_local[median_index+1:],reverse=True,key=lambda x: x[0])
		    #cv2.circle(blank_image,  (tmp_x,tmp_y),tmp_r, GREEN , 2)	
		#cv2.imshow('thre_image', tmp_thresh)
		#cv2.imshow('blank_image', blank_image)
		#cv2.waitKey(0)		
	    else:
		if( cover_list == []):
		    cover_list.append( (each_circle[0] , each_circle[1] ) )
		    cv2.circle( image_ori , each_circle[1] , each_circle[0] , each_circle[4], 2 )
		    cv2.drawContours(image_color_part[ each_circle[3][0] : each_circle[3][1] , each_circle[3][2] : each_circle[3][3] ],each_circle[2],0,each_circle[5],-1)
		    rice_number += 1
		    #cv2.imshow('Watershed-part:'+str(N), cv2.resize(image_ori[each_circle[3][0] : each_circle[3][1] , each_circle[3][2] : each_circle[3][3]], (0,0), fx=0.3, fy=0.3))
		    #cv2.imshow('Watershed:'+str(N), cv2.resize(image_ori, (0,0), fx=0.3, fy=0.3))
		    #cv2.waitKey(0)		    
		else:
		    over_cover = 0
		    if(each_circle[0]<0.5*mean_r):
			continue
		    for c in cover_list:  
			if( pow(each_circle[1][0]-c[1][0],2)+pow(each_circle[1][1]-c[1][1], 2) < pow(min(each_circle[0],c[0]),2) ):
			    over_cover = 1	
			    break
		    if(not over_cover):
			cover_list.append((each_circle[0],each_circle[1]))
			cv2.circle(image_ori,  each_circle[1], each_circle[0], each_circle[4], 2)
			cv2.drawContours(image_color_part[ each_circle[3][0] : each_circle[3][1] , each_circle[3][2] : each_circle[3][3] ],each_circle[2],0,each_circle[5],-1)
			rice_number += 1
			#print each_circle[3][0] , each_circle[3][1] , each_circle[3][2] , each_circle[3][3] , each_circle[1],each_circle[0]
			#cv2.imshow('Watershed-part:'+str(N), cv2.resize(image_ori[each_circle[3][0] : each_circle[3][1] , each_circle[3][2] : each_circle[3][3]], (0,0), fx=0.3, fy=0.3))
			#cv2.imshow('Watershed:'+str(N), cv2.resize(image_ori, (0,0), fx=0.3, fy=0.3))
			#cv2.waitKey(0)
		
		
	
	##circle.sort()
	##mean_r = circle[(len(circle)+1)/2][0]
	##print 'mean_r:' + str(mean_r)
	###( r , (x+shift_h, y+shift_v) , cnts , (shift_v , shift_v + windowSize_v ,  shift_h , shift_h + windowSize_h) )
	##for c in circle:
	    ##if(c[0] <= mean_r):
		##cv2.circle( image_ori , c[1] , c[0] , circle_color, 2 )
		##continue
	    ##if(c[0]<mean_r*1.2 ):
		###print c[0], c[1], circle_color
		##cv2.circle( image_ori , c[1] , c[0] , circle_color, 2 )
	    ##else:
		##N-=1
		##if(c[0]>=1.2*mean_r):
		    
		    ##cv2.circle(image_ori, c[1], c[0],  RED, 2)
		    ##blank_image = np.zeros((windowSize_v,windowSize_h,3), np.uint8)
		    ##blank_image[:,:] = BLACK
		    ##cv2.drawContours(blank_image,c[2],0, WHITE ,-1)
		    ###print type(blank_image),type(image_ori)
		    ###cv2.imshow('blank_image', blank_image)
		    ###cv2.waitKey(0)
		    ##tmp_gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
		    ##tmp_thresh = cv2.threshold(tmp_gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]		    
		    ##tmp_D = ndimage.distance_transform_edt(tmp_thresh)
		    ##tmp_localMax = peak_local_max(tmp_D, indices=False, min_distance=20,labels=tmp_thresh)			    
		    ##tmp_markers = ndimage.label(tmp_localMax, structure=np.ones((3, 3)))[0]
		    ##tmp_labels = watershed(-tmp_D, tmp_markers, mask=tmp_thresh)
		    ###print 'big circle'
		    ##tmp_clist = []
		    ##for tmp_label in np.unique(tmp_labels):
			##tmp_mask = np.zeros(tmp_gray.shape, dtype="uint8")
			##tmp_mask[tmp_labels == tmp_label] = 255
			##tmp_cnts = cv2.findContours(tmp_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
			##tmp_c = max(tmp_cnts, key=cv2.contourArea)
			##((tmp_x, tmp_y), tmp_r) = cv2.minEnclosingCircle(tmp_c)
			##tmp_x = int(tmp_x)
			##tmp_y = int(tmp_y)
			##tmp_r = int(tmp_r)
			##tmp_clist.append((tmp_r,(tmp_x,tmp_y),c[3][0] , c[3][1] , c[3][2] , c[3][3]))
			##cv2.circle(image_ori[ c[3][0] : c[3][1] , c[3][2] : c[3][3] ],  (tmp_x,tmp_y),tmp_r,  BLUE, 2)
		    #tmp_clist.sort(reverse=True)
		    #distance_clist = []
		    #for t in tmp_clist[1:]:
			#if(distance_clist == []):
			    #distance_clist.append((t[0],t[1]))
			    #cv2.circle(image_ori[ t[2] : t[3] , t[4] : t[5] ],  t[1],t[0],  BLUE, 2)
			#else:
			    #for d in  distance_clist:
				#if( pow(t[1][0]-d[1][0],2)+pow(t[1][1]-d[1][1], 2) < pow(min(t[0],d[0]),2) ):
				    #over_cover = 1	
				    #break
			    #if(not over_cover):
				#N+=1
				#distance_clist.append((t[0],t[1]))
				#cv2.circle(image_ori[ t[2] : t[3] , t[4] : t[5] ],  t[1],t[0],  BLUE, 2)
			
			
    
    elapsed_time = time.time() - start_time
    
    #print 'PyrMeanShift_time: {} s'.format(PyrMeanShift_time)
    #print 'Color2gray_time: {} s'.format(Color2gray_time)
    #print 'Localmaxmin_time: {} s'.format(Localmaxmin_time)
    #print 'Threshold_time: {} s'.format(Threshold_time)
    #print 'Mix_time: {} s'.format(Mix_time)
    #print 'Draw_circle_time: {} s'.format(Draw_circle_time)
   
    #total_rice_number = len(r_list_local)
    print 'rice number:'+ str(rice_number)    
    print 'Totle elapsed_time: {} s'.format(elapsed_time)    

    if(TF_show):
	cv2.imshow('Wstershed:'+str(rice_number), cv2.resize(image_ori, (0,0), fx=0.3, fy=0.3))
	cv2.imshow('Binary:'+str(rice_number), cv2.resize(thresh, (0,0), fx=0.3, fy=0.3))
	cv2.imshow('Color_part:'+str(rice_number), cv2.resize(image_color_part, (0,0), fx=0.3, fy=0.3))
	
	#cv2.imshow('Wstershed:'+str(rice_number), image_ori)
	#cv2.imshow('Binary:'+str(rice_number), thresh)	
	cv2.waitKey(0)    
    
    if(TF_write):
	cv2.imwrite(output_path+fileName[:-4]+'-['+ str(rice_number) +']-sliWin-'+str(n_part)+'-part.jpg',image_ori)
	cv2.imwrite(output_path+fileName[:-4]+'-['+ str(rice_number) +']-binary-'+str(n_part)+'-part.jpg',thresh)
	cv2.imwrite(output_path+fileName[:-4]+'-['+ str(rice_number) +']-Color_part-'+str(n_part)+'-part.jpg',image_color_part)
    
    print fileName+" finished!" +' ['+ str(rice_number) +']\n--------------------------------\n'
    del image_gray,image_ori,t2_img,tmp,tmp_img,gray,shifted,dele

print "All finished!!"  


def global_equalization(image):
    (B_Channel, G_Channel, R_Channel) = cv2.split(image)
    B = cv2.equalizeHist(B_Channel)
    G = cv2.equalizeHist(G_Channel)
    R = cv2.equalizeHist(R_Channel)
    return cv2.merge((B,G,R))
    


def remove_max_backgroung():
    for part_v in range(n_part):
	for part_h in range(n_part):

	    print part_v,part_h
	    #if((part_v,part_h)!=(2.0,2.0)):
		#continue

	    shift_v = int(part_v*(height_ori/n_part))
	    windowSize_v = int(height_ori/n_part)
	    shift_h = int(part_h*(width_ori/n_part))
	    windowSize_h = int(width_ori/n_part)

	    image = tmp_img[shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h]
	    height,width,channel = image.shape

	    shifted[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ] = cv2.pyrMeanShiftFiltering(image, 21, 51)	    	
	    gray = cv2.cvtColor(shifted[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ], cv2.COLOR_BGR2GRAY)

	    shift = shifted[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ].copy()

	    he,wi,ch = shift.shape
	    #gray = np.zeros((height_ori,width_ori), np.float32)
	    #for h in range(he):
		#for w in range(wi):
		#(B_, G_, R_) = cv2.split(shift)
		#gray[h,w] = (np.float32)(R_[h,w]*0.299 + G_[h,w]*0.587 + B_[h,w]*0.114)
	    (B, G, R) = cv2.split(shift)
	    gray_float = (B*0.114 + G*0.587+R*0.299)
	    #print shift
	    #cv2.imshow('gray', shift)
	    #cv2.waitKey(0)

	    gray_list = []
	    for h in range(he):
		for w in range(wi):
		    gray_list.append((gray_float[h,w],h,w))
	    gray_list.sort()
	    #print gray_list
	    tmp = 300
	    div = 0

	    for t in gray_list:
		if(tmp==300):
		    tmp = t[0]
		    continue
		else:
		    if(tmp != t[0]):
			div+=1
		    tmp = t[0]

	    mean_differ = ( gray_float.max()-gray_float.min() )/div
	    print mean_differ
	    head = 0
	    diff_list = []
	    mp = gray_list[0][0]
	    for i in range(len(gray_list)-1):
		if(mp!=gray_list[i][0]):
		    #print gray_list[i+1][0],gray_list[i][0] , mean_differ
		    mp=gray_list[i][0]
		if( gray_list[i+1][0]-gray_list[i][0] > mean_differ ):
		    diff_list.append((i-head,head,i))
		    head = i
	    diff_list.sort()
	    #if((part_v,part_h)==(2.0,2.0)):
		#print mean_differ,diff_list,gray.max(),gray.min(),div
		#cv2.imshow('gray', gray)    
		#cv2.waitKey(0)		
	    if(len(diff_list)==0):
		print '%.1f , %.1f all background!' % (part_v,part_h)
		for thr in thresh[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ]:
		    thr = 0
		continue
	    #cv2.imshow('gray', gray)
	    #cv2.waitKey(0)

	    ran = diff_list[len(diff_list)-1]
	    for i in range(ran[1],ran[2]):
		gray[ gray_list[i][1],gray_list[i][2] ] = 0
	    #cv2.imshow('gray', gray)
	    #cv2.waitKey(0)

	    thresh[ shift_v : shift_v + windowSize_v ,  shift_h : shift_h + windowSize_h ] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	    #cv2.imshow('thresh', cv2.resize(thresh, (0,0), fx=0.3, fy=0.3))
	    #cv2.waitKey(0)		
    cv2.imshow('thresh-all',cv2.resize(thresh, (0,0), fx=0.3, fy=0.3) )
    #cv2.imshow('thresh-all',thresh )
    cv2.waitKey(0) 
