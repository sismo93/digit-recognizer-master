import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop ,SGD

sns.set(style='white', context='notebook', palette='deep')



# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd_optimizer = SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, 
)


def data_prepare(train_file ,test_file ,img_rows , img_cols , num_classes):
	train_data  = pd.read_csv(train_file)
	test_data  = pd.read_csv(test_file)
	out_y  = to_categorical(train_data.label , num_classes)
	num_images = train_data.shape[0]
	x_input = train_data.values[: , 1:]
	test_data = test_data.values[ 1:, :]
	x_input_reshaped  = x_input.reshape(num_images , img_rows , img_cols , 1)
	test_reshaped  = test_data.reshape(-1 , img_rows , img_cols , 1)
	out_x = x_input_reshaped/ 255
	test_reshaped = test_reshaped /255  
	return out_x  , out_y , test_reshaped





def build_sequential_model( img_rows , img_cols ,num_classes) :

	model = Sequential()
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
									 activation ='relu', input_shape = (img_rows,img_cols,1)))
	model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
									 activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
									 activation ='relu'))
	model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
									 activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation = "softmax"))

	return model


# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
							normalize=False,
							title='Confusion matrix',
							cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, cm[i, j],
								 horizontalalignment="center",
								 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()