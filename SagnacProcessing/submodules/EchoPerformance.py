#!/usr/bin/env python
# coding: utf-8


def  __echo_performance(t1, t2):
    
    '''
    prints formated performance time 
    
    t1:  starttime
    t2:  endtime 
    
    '''
    
    dt = t2-t1
    
    if dt < 60:
        print(f"\n elapsed time: {round(dt,2)} sec")
    
    elif dt >= 60 and dt < 3600:
        print(f"\n elapsed time: {int(dt/60)} min {round(dt%60,2)} sec")
    
    elif dt >= 3600: 
        print(f"\n elapsed time: {int(dt/3600)} hr {int(dt%3600)} min {round(dt%60,2)} sec")
