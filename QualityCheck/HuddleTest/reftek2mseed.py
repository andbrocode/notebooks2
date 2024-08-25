#!/usr/bin/env python
# coding: utf-8

# # Reftek2MSEED

# #### Convert Reftek data to MSEED files 
# 

# Loading required libraries

# In[12]:


import obspy
import os
import numpy as np

import subprocess
import warnings
warnings.filterwarnings('ignore')


# Runs a list command in the shell to get content of directory

# In[9]:


def __get_content_list(path_in):
    p = subprocess.Popen(["ls", path_in], stdout=subprocess.PIPE)
    out, err = p.communicate()

    return str(out)[2:-1].split("\\n")[:-1]


# Required parameters are entered by user

# In[15]:


# project_path = "/home/brotzer/Desktop/Huddle_STS2/Huddle_Test_1/"
# reftek = "AC48"

# sn = 29422
# instrument = f"STS2_{sn}_all"


project_path = input("Enter project path: ")

if project_path[-1] == "/":
    project_path = project_path[:-1]

reftek = input("Enter recorder number (e.g. AC48): ")

sn = input("Enter serial number of instrument: ")

instrument = input("Enter instrument type (e.g. STS2): ")

channels = str(input("Enter assigned channels (e.g. 123): "))


# create a list of all days for which data has been recorded

# In[22]:


days_path = f"{project_path}/raw/{reftek}/"
# days = !ls $days_path

days = __get_content_list(days_path)


# Iteratively process recorded files of days and write daily files as mseed

# In[107]:


st = obspy.core.stream.Stream()


count_masked = 0;

for day in days:
    
    st = obspy.core.stream.Stream()

    files_path = f"{project_path}/raw/{reftek}/{day}/{reftek}/1/"
#     files = !ls $files_path
    files = __get_content_list(files_path)
    
    print("\nreading and merging data ...")
    for file in files:
        print(file)
        tr = obspy.read(f"{project_path}/raw/{reftek}/{day}/{reftek}/1/{file}",
                       network = "BW",
                       station = f"{sn}",
                       location = "",
                       );

        if channels == "123":
            for i in range(3):
                st.append(tr[i]);


        elif channels == "456":
            for i in range(3):
                st.append(tr[i]);
    

    
    st.merge()
    
    for i in range(3):
        if np.ma.isMaskedArray(st[i].data):
            st[i].data = np.ma.filled(st[i].data, fill_value=0.0)
            print(f"{st[i].stats.channel} unmasked!")
            count_masked += 1    
            
    opath = f"{project_path}/{instrument}_{sn}/{day}"
    obasename = f"{instrument}_{sn}"

    if not os.path.exists(opath):
        os.makedirs(opath)

#     print("\nset header info ...")
    
    
    print(f"\nwriting data to: \n {opath}/{obasename} ...")

    st.write(f"{opath}/{obasename}_all.mseed", format="MSEED")

    for i in range(3):
        st[i].write(f"{opath}/{obasename}_{st[i].stats.channel}.mseed", format="MSEED")

    del st    

    print(f"\n{count_masked} traces were unmasked!")
    
    print("\nDone")


# In[ ]:




