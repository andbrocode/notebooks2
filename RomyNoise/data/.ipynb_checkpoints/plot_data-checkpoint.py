import matplotlib.pyplot as plt
import numpy
import os
root = '/Users/yararossi/Documents/Work/Random_Workstuff/'

fig, ax = plt.subplots(1,2, figsize=(11,5), sharex=True)
for filename in os.listdir(root):
    if 'Mag' in filename:
        data = numpy.loadtxt('%s/%s' %(root,filename), delimiter = '\t')

        if '20km' in filename:
            liner = '-'
            color = 'k'
        elif '200km' in filename:
            liner = '-.'
            color = 'grey'
        elif '2000km' in filename:
            liner = '--'
            color = 'brown'
        ax[0].loglog(data[:,0], data[:,1], color=color, linestyle=liner)
        ax[1].loglog(data[:,0], data[:,2], color=color, linestyle=liner)

ax[0].set_ylabel('Acceleration [m/s/s]')
ax[1].set_ylabel('Rotation rate [rad/s]')
ax[0].set_xlabel('Frequency [Hz]')
ax[1].set_xlabel('Frequency [Hz]')
ax[0].grid('on')
ax[1].grid('on')
plt.show()
