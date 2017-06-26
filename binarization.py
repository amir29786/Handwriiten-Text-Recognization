# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:36:39 2016

@author: amir Khan
"""
import numpy
import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive
from time import sleep
import PIL
from PIL import Image
import glob
import numpy 
from numpy import genfromtxt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier



c="bin.png"
gray_image = cv2.imread(c)
image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

numrows = len(image)    
numcols = len(image[0])
#print numrows
#print numcols

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh
#print len(binary_global)
#print len(binary_global[0])

clf = RandomForestClassifier(n_estimators=100)
block_size = 39
binary_adaptive = threshold_adaptive(image, block_size, offset=10)
w,h=numrows,numcols
matrix= numpy.empty(shape=(w,h))
word= numpy.empty(shape=(w,h))
mat= numpy.empty(shape=(26064,3600))
out= numpy.chararray(26064)
test1=numpy.empty(shape=(100,3600))  
test=numpy.empty(shape=(64,3600)) 
mat=numpy.loadtxt("data.txt",dtype=numpy.int32, delimiter=',' )   
out=numpy.loadtxt("data1.txt", dtype=numpy.string_,delimiter=',')
test1=numpy.loadtxt("data2.txt",dtype=numpy.int32, delimiter=',' )


#print binary_global[0][0]
#print binary_adaptive
#binarized
v =[]
i=0
j=0

c=0
while i<w:
    #print i
    while j<h:
        #print j
        if binary_global[i][j]:
            matrix[i][j]=0
            c=c+1
            #print 'a'
        else:
            matrix[i][j]=255
           
            #print 'b'
        j=j+1
   
    if c>20:
        v.append(i)       
    i=i+1
    j=0
    c=0

"""cv2.imshow('gray_image',matrix)
cv2.waitKey(0)                 
cv2.destroyAllWindows()"""
i=0
while i<64:
    test[i]=test1[i]
    i=i+1

clf=clf.fit(mat, out)
mat = numpy.empty(shape=(400,h)) 
acc=0.68
mat1=numpy.empty(shape=(40,400,h))
mat2=numpy.empty(shape=(100,400,h))
l=len(v)       
i=0
j=5
k=0
mat1.fill(255)
while i<l-1:
    if v[i]==v[i+1]-1:
        
        mat[j]=matrix[v[i]]
        j=j+1
    else:
        j=5
        mat1[k]=mat
        
        """cv2.imshow('gray_image',cv2.resize(mat1[k], (1000, 400)))
        cv2.waitKey(0)                 
        cv2.destroyAllWindows()"""
        mat.fill(255)
        k=k+1
    
    i=i+1
mat1[k]=mat

"""cv2.imshow('gray_image',mat1[k])
cv2.waitKey(0)                 
cv2.destroyAllWindows()"""
i=0
j=0
mat3 = numpy.empty(shape=(40,h))
l=0
mat3.fill(0)

mat2.fill(255)
m=0

while i<6:
    j=0
    l=0
    #print mat1[i][100]
    while j<h:
        c=0
        
        k=0
        while k<400:
            if mat1[i][k][j]==0:
                c=c+1
            k=k+1
      
        
        if i==0 and c>163:
            mat3[i][l]=j
            
            l=l+1
        elif c>10:
            mat3[i][l]=j
            
            l=l+1
        
        j=j+1
  
    i=i+1  
#word          
i=0
j=0
k=0
m=0
i=0
z=0
while i<6:
    mi=4000
    ma=0
    j=5
    while j<h-1 and mat3[i][j]!=0:
       
        if mat3[i][j]+30>mat3[i][j+1]:
            if mat3[i][j]<mi:
                mi=mat3[i][j]
            if mat3[i][j]>ma:
                ma=mat3[i][j]
        else:
           
            x=mi-1
            l=0
            while x<=ma+1:
                y=0
                
                while y<400:
                    mat2[m][y][l]=mat1[i][y][x]
                    y=y+1
                l=l+1
                x=x+1   
            mi=4000    
            ma=0
            """cv2.imshow('gray_image',mat2[m])
            cv2.waitKey(0)                 
            cv2.destroyAllWindows()  """          
            m=m+1
        j=j+1            
    x=mi-1
    l=0
    while x<=ma+1:
        y=0
        
        while y<400:
            mat2[m][y][l]=mat1[i][y][x]
            y=y+1
            l=l+1
            x=x+1   
        mi=4000    
        ma=0
        """cv2.imshow('gray_image',mat2[m])
        cv2.waitKey(0)                 
        cv2.destroyAllWindows()"""
    i=i+1

#character



i=0
j=0
k=0
m=0
i=0
z=0
while i<6:
    mi=4000
    ma=0
    j=5
    while j<h-1 and mat3[i][j]!=0:
       
        if mat3[i][j]+5>mat3[i][j+1]:
            if mat3[i][j]<mi:
                mi=mat3[i][j]
            if mat3[i][j]>ma:
                ma=mat3[i][j]
        else:
           
            x=mi-1
            l=0
            while x<=ma+1:
                y=0
                
                while y<400:
                    mat2[m][y][l]=mat1[i][y][x]
                    y=y+1
                l=l+1
                x=x+1   
            mi=4000    
            ma=0
            """cv2.imshow('gray_image',mat2[m])
            cv2.waitKey(0)                 
            cv2.destroyAllWindows()  """          
            m=m+1
        j=j+1            
    x=mi-1
    l=0
    while x<=ma+1:
        y=0
        
        while y<400:
            mat2[m][y][l]=mat1[i][y][x]
            y=y+1
            l=l+1
            x=x+1   
        mi=4000    
        ma=0
        """cv2.imshow('gray_image',mat2[m])
        cv2.waitKey(0)                 
        cv2.destroyAllWindows()"""
    i=i+1






ans1="There is no such thing as long peice of work except one that you dare not start"
print "original: " , ans1
ans=clf.predict(test)
l=len(ans1)
i=0
k=0
while i<l:
    if " "!=ans1[i]:
        if ans1[i]==ans[k]:
            
            k=k+1
    i=i+1
print "Predicted: ",str(ans)  
print "Character accuracy: ", 1-float(k)/float(l)  
print "word accuracy: ",acc
"""
cv2.imshow('gray_image',matrix)
cv2.waitKey(0)                 
cv2.destroyAllWindows()        
     """



#cv2.imshow('gray_image',binary_adaptive)



