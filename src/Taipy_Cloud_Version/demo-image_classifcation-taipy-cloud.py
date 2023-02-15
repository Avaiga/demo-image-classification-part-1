import tensorflow as tf
from tensorflow.keras import layers, models  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.utils import to_categorical  
import pandas as pd   
import matplotlib.pyplot as plt  
import numpy as np
from PIL import Image 

class_names =  ['AIRPLANE', 'AUTOMOBILE', 'BIRD', 'CAT', 'DEER', 'DOG', 'FROG', 'HORSE', 'SHIP', 'TRUCK']

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train / 255.0
y_train = to_categorical(y_train, len(class_names))

x_test = x_test / 255.0
y_test = to_categorical(y_test, len(class_names))

#########################################################################################################
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

df = pd.read_csv("saved_models/df.csv")
df["N_Epochs"] = range(1,len(df)+1)



#STATE VARIABLES 

model  = None

# Parameters for models & training
epochs = 1
input_model_name = "model" 

# Parameters for trained model
trained_model_path = ""

# Parameters for CIFAR dataset
cifar_image_index  = 10
cifar_image_path = "images/sample/taipy.jpg"
cifar_predicted_label = 'NA'
cifar_true_label = 'NA'

# Parameters for online image
online_image_url  = "URL"
online_image_path = "images/sample/airplane.jpg" 
online_image_count = 0
online_image_predicted_label  = 'NA' # predicted label for the online image



#P1
from taipy import Gui
from taipy.gui import invoke_long_callback, notify
import urllib

p1 = """
<center><h1>Image Classification CNN</h1></center>

<|layout|columns=1 3|
<|
## PARAMETERS
Enter chosen optimal numper of epochs:   
<|{epochs}|input|>  


Register model name:   
<|{input_model_name}|input|>

Train the model with the Training + Validation sets:  
<|START TRAINING|button|on_action=train_button|>  

### Upload Trained Model
<|{trained_model_path}|file_selector|label=Upload trained model|on_action=load_trained_model|extensions=.h5|>
|>

<|
<center><h2> Val_loss and Accuracy </h2></center>
<|{df}|chart|x=N_Epochs|y[1]=accuracy|y[2]=val_accuracy|>
|>
|>
___
"""

def merged_train(model,number_of_epochs,name):
    # merge the training and validation sets
    #x_all = np.concatenate((x_train, x_test))
    #y_all = np.concatenate((y_train, y_test))

    # train with the merged dataset
    #history = model.fit(
    #    datagen.flow(x_all, y_all, batch_size=64),
    #    epochs=number_of_epochs)

    #model.save("saved_models/{}.h5".format(name),save_format='h5')
    print("TRAINING & SAVING COMPLETED!")

def train_button(state):
    notify(state, "info", "Started training model with {} epochs".format(state.epochs), True, 1000)
    #model = create_model()
    invoke_long_callback(state,merged_train,[model, int(state.epochs), state.input_model_name])

def load_trained_model(state):
    loaded_model = tf.keras.models.load_model(state.trained_model_path)
    state.model = loaded_model


#Second half of the applications
p2 = """ 
<|layout|columns=1 3|
<|
### CIFAR10 Images Prediction
Enter CIFAR10 image index:  |

<|{cifar_image_index}|input|>  
<|PREDICT CIFAR IMAGE|button|on_action=predict_cifar_image|>

<|{cifar_image_path}|image|height=100px|width=100px|>

##Predicted label: <|{cifar_predicted_label}|>  
##True label: <|{cifar_true_label}|>

|>

<|
###Paste an online image link here for prediction:  

<|{online_image_url}|input|on_action=load_online_image|>  

<center> <|{online_image_path}|image|height=300px|width=300px|> </center>

<|PREDICT ONLINE IMAGE|button|on_action=predict_online_image|>

## Predicted label: <|{online_image_predicted_label }|>
|>
|>
"""

def predict_cifar_image(state):
    #Retrieve the cifar image at the specified index and save as PIL Image obj
    cifar_img_idx = int(state.cifar_image_index )
    cifar_img_data = x_test[cifar_img_idx]
    cifar_img = Image.fromarray(np.uint8(cifar_img_data*255))
    cifar_img.save("images/cifar10_saved/{}.jpg".format(cifar_img_idx))

    #Predict the label of the CIFAR image
    img_for_pred = np.expand_dims(x_test[cifar_img_idx], axis=0)
    cifar_img_pred_label = np.argmax(state.model.predict(img_for_pred))
    cifar_img_true_label = y_test[cifar_img_idx].argmax() 
    
    #Update the GUI
    state.cifar_image_path = "images/cifar10_saved/{}.jpg".format(cifar_img_idx)
    state.cifar_predicted_label = str(class_names[cifar_img_pred_label])
    state.cifar_true_label = str(class_names[cifar_img_true_label])

def load_online_image(state):
    urllib.request.urlretrieve(state.online_image_url, "images/online_image.jpg")
    state.online_image_path = "images/online_image.jpg"

def predict_online_image(state):
    #Retrieve & save online image in order to show on the image box
    urllib.request.urlretrieve(state.online_image_url , "images/saved_images/{}.jpg".format(state.online_image_count))
    state.online_image_path = "images/saved_images/{}.jpg".format(state.online_image_count)

    #Predict the label of the online image
    img_array = tf.keras.utils.load_img(state.online_image_path, target_size=(32, 32))
    image = tf.keras.utils.img_to_array(img_array)  # (height, width, channels)
    image = np.expand_dims(image, axis=0) / 255.    # (1, height, width, channels) + normalize

    #Update the GUI
    state.online_image_predicted_label  = class_names[np.argmax(state.model.predict(image))]
    state.online_image_count += 1

Gui(page=p1+p2).run(dark_mode=False)
