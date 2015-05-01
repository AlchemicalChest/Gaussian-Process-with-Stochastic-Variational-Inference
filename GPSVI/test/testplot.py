import time
import numpy as np
from matplotlib import pyplot as plt

 
plt.ion() # set plot to animated
ydata = [0]*50
 
# make plot
line, = plt.plot(ydata)
plt.ylim([10,40])
 
# start data collection
while True:  
    y = np.random.rand()
    plt.ylim([0, np.max(y)+5])
    ydata.append(y)
    del ydata[0]
    line.set_xdata(range(len(ydata)))
    line.set_ydata(ydata)  # update the data
    plt.draw() # update the plot
    time.sleep(0.05)
    