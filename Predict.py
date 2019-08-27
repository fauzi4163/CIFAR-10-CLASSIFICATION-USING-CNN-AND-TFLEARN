# predict.py

import TFLearnDNN as net
import random 
import Class_Info
from tflearn.datasets import cifar10
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np



model = net.model

path_to_model = 'model.tfl'

(X, Y), (X_test, Y_test) = cifar10.load_data()

model.load(path_to_model)

# Randomly take an image from the test set
rand_index = random.randint(0, len(X_test) - 1)
print("Random Number : ",rand_index)

x = X_test[rand_index].reshape((32, 32, 3))  
result = model.predict([x])[0] # Predict
print(result)
prediction = result.tolist().index(max(result)) # The index represents the number predicted in this case


#Calling objects from Class_Info file
Class_Info.data_path = ""

class_names = Class_Info.load_class_names()
print("Prediction: ",class_names[prediction])
print(prediction)



#Printing the Image
f = open('cifar-10-batches-py/test_batch','rb')


tupled_data= pickle.load(f, encoding='bytes')

f.close()

img = tupled_data[b'data']

single_img = np.array(img[rand_index])

#single_img_reshaped = single_img.reshape(32,32,3)
single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))

plt.imshow(single_img_reshaped)
plt.show()
