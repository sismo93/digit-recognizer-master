from neural_network_lib import nn 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# load data
train_path = "../digit-recognizer/data/train.csv"
test_path = "../digit-recognizer/data/test.csv"


img_rows , img_cols = 28 , 28
num_classes = 10 

#load data from file using panda
#Categorizing the final class/the last layer
#Reshape the matrix with adding one dimention (28*28*1)
out_x  , out_y , test_data  = nn.data_prepare(train_path ,test_path, img_rows,img_cols, num_classes)
# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(out_x, out_y, test_size = 0.1, random_state=random_seed)

#print ("the out_x and the out_Y " , out_x , out_y)

model = nn.build_sequential_model( img_rows , img_cols ,num_classes)



#Compiling the model

model.compile(optimizer= "adam", loss = "categorical_crossentropy", metrics=["accuracy"])



datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
		zoom_range = 0.1, # Randomly zoom image 
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False)  # randomly flip images



# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
											patience=3, 
											verbose=1, 
											factor=0.5, 
											min_lr=0.00001)


datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=86),
							  epochs = 2, validation_data = (X_val,Y_val),
							  verbose = 1, steps_per_epoch=X_train.shape[0] // 86
							  , callbacks=[learning_rate_reduction])


#Fit without data augmentation
#history = model.fit(X_train, Y_train, batch_size = 128, epochs = 17, 
#        validation_data = (X_val, Y_val), verbose = 2)

# Evaluate the model
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()
# save model
model.save('../digit-recognizer/user_interface/hand_written.h5')



#Confusion matrix
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
nn.plot_confusion_matrix(confusion_mtx, classes = range(10)) 


plt.show()

# predict results
results = model.predict(test_data)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_result_datagen.csv",index=False)
