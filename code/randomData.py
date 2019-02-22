# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:32:29 2019

@author: sun
"""

'''
生成随机数数据
'''
import random

def generateOneData():
    
    item = []
    
    b_tgt = random.randint(0,1)
    item.append(b_tgt)
    
    level = ['X','Y','Z']
    cat_input1 = level[random.randint(0,2)]
    item.append(cat_input1)
    
    val_lev = ['A','B','C','D','E']
    cat_input2 = val_lev[random.randint(0,4)]
    item.append(cat_input2)
    
    rfm1 = round(random.uniform(0,40),2)
    item.append(rfm1)
    rfm2 = round(random.uniform(0,40),2)
    item.append(rfm2)
    rfm3 = round(random.uniform(0,50),2)
    item.append(rfm3)
    rfm4 = round(random.uniform(10,50),2)
    item.append(rfm4)
    rfm5 = random.randint(0,15)
    item.append(rfm5)
    rfm6 = random.randint(0,20)
    item.append(rfm6)
    rfm7 = random.randint(0,10)
    item.append(rfm7)
    rfm8 = random.randint(0,20)
    item.append(rfm8)
    rfm9 = random.randint(0,50)
    item.append(rfm9)
    rfm10 = random.randint(0,20)
    item.append(rfm10)
    rfm11 = random.randint(0,15)
    item.append(rfm11)
    rfm12 = random.randint(0,150)
    item.append(rfm12)
    
    demog_age = random.randint(20,50)
    item.append(demog_age)
    agents = ['男','女']
    demog_agent = agents[random.randint(0,1)]
    item.append(demog_agent)
    hos = ['是','否']
    demog_ho = hos[random.randint(0,1)]
    item.append(demog_ho)
    demog_homeval = round(random.uniform(10000,30000),2)
    item.append(demog_homeval)
    demog_inc = round(random.uniform(10000,20000),2)
    item.append(demog_inc)
    demog_pr = round(random.random(),2)
    item.append(demog_pr)
    
    return item


if __name__ =="__main__":
    
    items = []
    
    for i in range(1000):
        items.append(generateOneData())
    
    
    f = open(r'C:\Users\sun\Desktop\finance\data.txt','w')       
    
    for item in items:
        
        res = ''
        
        for i in item:
            res+=str(i)+","
        
        f.write(str(res)+"\n")
        
    f.close()    
    
    