import pandas as pd
import os
import shutil

#covid patient xray
file_path="covidxray/covid-chestxray-dataset-master/metadata.csv"
IMAGE_path="covidxray/covid-chestxray-dataset-master/images"
df=pd.read_csv(file_path)
#print(df.shape)
#df.head()

target_dir="dataset/covid"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
    #print("covid folder created")

count=0
for (i,row) in df.iterrows():
    if row["finding"]=="COVID-19" and row["view"]=="PA":
        filename=row["filename"]
        image_path= os.path.join(IMAGE_path,filename)
        image_copy_path=os.path.join(target_dir,filename)
        shutil.copy2(image_path,image_copy_path)
        #print("moving image ",count)
        count += 1
#print(count)



#sample of normal patient xray
import random
normal_file_path="normalxray/chest_xray/train/NORMAL"
normal_target_dir="dataset/normal"
image_names=os.listdir(normal_file_path)

for i in range(201):
    image_name=image_names[i]
    image_path=os.path.join(normal_file_path,image_name)
    target_path=os.path.join(normal_target_dir,image_name)
    shutil.copy2(image_path,target_path)
    #print("copied image ",i)

CNN model for covid-19 detection
Code:
	#CNN model

train_path="dataset/train"
test_path="dataset/test"

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
model.summary()

# Train from scratch
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)
test_dataset = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = (224,224),
    batch_size = 32,
class_mode = 'binary')
train_generator.class_indices

validation_generator = test_dataset.flow_from_directory(
    'dataset/test',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps=2
)

y_actual=[]
y_test=[]
import os
train_generator.class_indices
for i in os.listdir("./dataset/test/normal/"):
    img=image.load_img("./dataset/test/normal/"+i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    p=model.predict_classes(img)
    y_test.append(p[0,0])
    y_actual.append(1)
for i in os.listdir("./dataset/test/covid/"):
    img=image.load_img("./dataset/test/covid/"+i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    p=model.predict_classes(img)
    y_test.append(p[0,0])
    y_actual.append(0)
y_actual=np.array(y_actual)
y_test=np.array(y_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_actual,y_test)
import seaborn as sns
sns.heatmap(cm,cmap="plasma",annot=True)
