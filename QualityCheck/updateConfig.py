#!/usr/bin/env python
# coding: utf-8

# In[6]:


from configparser import ConfigParser
import sys


name = sys.argv[1]

path = sys.argv[2]
code = sys.argv[3]
year = sys.argv[4]


conf= ConfigParser()
conf.read(name)

var = conf["VAR"]

var['opath'] = path
var['code'] = code
var['year'] = year

with open(name, 'w') as f:
    conf.write(f)


# In[ ]:




