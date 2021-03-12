from keras.models import load_model
import cv2
import matplotlib.pyplot as plt

classes=[0,1,2,3,4,5,6,7,8,9]
model=load_model('hand_written.h5')




def predicting():
	img=cv2.imread('image.png',0)
	img=cv2.bitwise_not(img)
	img=cv2.resize(img,(28,28))  
	img=img.reshape(-1,28,28,1)
	img=img.astype('float32')
	img=img/255.0
	pred=model.predict(img)
	return pred

