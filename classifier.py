import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
X=np.load("image.npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())
classes=["A","B","C","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)
xTrain,xTest,yTrain,yTest=train_test_split(X,y,train_size=3500,test_size=1500,random_state=9)
xTrainScaled=xTrain/255.0
xTestScaled=xTest/255.0
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xTrainScaled,yTrain)
def getPrediction(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert("L")
    image_bw_resized=image_bw.resize((28,30),Image.ANTIALIAS)
    pixelFilter=20
    minPixel=np.percentile(image_bw_resized,pixelFilter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized-minPixel,0,255)
    maxPixel=np.max(image_bw_resized)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/maxPixel
    testSample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    testPred=clf.predict(testSample)
    return testPred[0]
