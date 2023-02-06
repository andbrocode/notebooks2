#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ctypes

# A. Create library
C_library = ctypes.CDLL("/home/brotzer/notebooks/test/clib.so")

# B. Specify function signatures
hello_fxn = C_library.say_hello
hello_fxn.argtypes = [ctypes.c_int]

# C. Invoke function
num_repeats = 5
hello_fxn(num_repeats)


# In[6]:


C_library = ctypes.CDLL("/home/brotzer/notebooks/test/hilbert.so")


# In[ ]:




