from __future__ import division
import pandas as pd


with open('model/modelUV.pkl') as f:
  myU, myV = pickle.load(f)
myR_pred = np.around(myU.dot(myV),2)


