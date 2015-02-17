from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model

net = load_model('main_net.pickle')
img = misc.imread("image.png")
print img
X = (img/255.0).astype(np.float32).reshape(-1, 1, 96, 96)
#print X
y = (net.predict(X)[0])*48+48
print y
plt.imshow(img, cmap=plt.cm.gray)
plt.scatter(y[0::2], y[1::2], marker='x', s=10)
plt.show()
