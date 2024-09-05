#!/usr/bin/env python
# coding: utf-8

# In[70]:

import numpy as np
import pyvisa
import time
import datetime
import os


# In[71]:


def store(_out, _filename):

    with open(_filename, "a") as file:
        file.write(_out)


# In[74]:


def main():

    try:
        rm = pyvisa.ResourceManager()
        # print(' -> available ressources:', rm.list_resources())

        # connect via ETH
        # inst = rm.open_resource('TCPIP0::10.5.5.5::inst0::INSTR')
        inst = rm.open_resource('TCPIP0::10.153.82.213::inst0::INSTR')

        # connect via USB
        # inst = rm.open_resource('USB::0x0AAD::0x0119::022019943::INSTR')

        inst.read_termination = '\n'

        inst.write_termination = '\n'

        # Query the Identification string
        print(' -> connected to: ', inst.query('*IDN?'))

        # Clear instrument io buffers and status
        inst.clear()

        inst.write('GTR')

        inst.write('SYSTem:DISPlay:UPDate ON')

        # inst.write('*RST') # Reset device
        # inst.write('INIT:CONT OFF')  # Switch OFF the continuous sweep

    except:
        print(f" -> not connected")

    tmax_sec = 3*86400

    # get starttime as utc
    starttime = datetime.datetime.now()

    elapsed_time = 0

    # specify filename
    path = "/home/pi/"
    filename = path+f"romy_fsr_{starttime.strftime('%Y%m%dT%H%M%S')}.csv"

    # create file with header if not existent
    if not os.path.isfile(filename):
        f = open(filename, "w")
        f.write("datetime,x_hz,y_dbm\n")
        f.close()

    while elapsed_time < tmax_sec:

        # get data
        _t = datetime.datetime.utcnow()
        try:
            _x = inst.query_ascii_values('CALC:MARK:FUNC:FPE:X?')
            _y = inst.query_ascii_values('CALC:MARK:FUNC:FPE:Y?')
        except:
             _x, _y = np.nan, np.nan

        # compute elapsed time since start
        _delta = datetime.datetime.now() - starttime
        elapsed_time = _delta.seconds

        # form output string
        out = f"{_t},{_x[0]},{_y[0]}\n"

        # write output
        store(out, filename)

        # Wait for 1 second
        time.sleep(1)

if __name__ == "__main__":
    main()


# In[ ]:




