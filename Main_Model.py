import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

keras = tf.keras
(train_ , vallidation_ , test_) , info = tfds.load('cats_vs_dogs' , split = ['train[:80%]' , 'train[80%:90]' , 'train[90%:]'] , with_info = True , as_supervised=True)

# let's look at the data 

total_train = 0
total_vall = 0
total_test = 0

for a in train_ : 
    total_train += 1


for a in vallidation_ : 
    total_vall += 1


for a in test_ : 
    total_test += 1

print(f'{total_train} train images about cats and dog')

print(f'{total_vall} vallidation images about cats and dog')

print(f'{total_test} test images about cats and dog')

label = info.features['label'].names

# let's look at the images

plt.figure(figsize = (4,4))
for image , lbl in train_.take(4) : 
    
    plt.imshow(image)
    plt.title(label[lbl])
    plt.colorbar()
    plt.grid(True)

    plt.show()

def formating(img , lbl) : 
    img_size = 150
    img = tf.cast(img , tf.float32) / 255.0
    img = tf.image.resize(img , (img_size , img_size))

    return img , lbl
batch_size = 32
train_ = train_.map(formating)
vallidation_ = vallidation_.map(formating)
test_ = test_.map(formating)

# let's look at the images

for image , lbl in train_.take(4) : 
    plt.figure(figsize = (4,4))
    plt.imshow(image)
    plt.title(label[lbl])
    plt.colorbar()
    plt.grid(True)

    plt.show()

# making better data

shuffle = 1000
batches = 32

train_ = train_.shuffle(shuffle).batch(batches)
vallidation_ = vallidation_.shuffle(shuffle).batch(batches)
test_ = test_.batch(batches)

# making our model with mobilnet v2 

base_layer = keras.applications.MobileNetV2(input_shape = (150,150,3) , weights = 'imagenet' , include_top=False)
avarage_pool = keras.layers.GlobalAveragePooling2D()
predition_layer = keras.layers.Dense(1)

base_layer.trainable = False

base_layer.summary()

for image , _ in train_.take(1) : 
    pass

plt.figure(figsize = (4 , 4))

for value in range(25) : 
    plt.subplot(5,5,value+1)
    plt.imshow(base_layer(image)[0][: , : , value])
    plt.xticks([])
    plt.yticks([])

plt.show()
     

# base model

model = keras.Sequential([
    base_layer , 
    avarage_pool , 
    predition_layer
])


model(image)

plt.plot(model(image))

plt.show()


lrt = 0.0001
op = keras.optimizers.RMSprop(lr = lrt)
loss = keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer = op , loss = loss , metrics = ['accuracy'])

model.summary()

model.fit(train_ , epochs=3 , validation_data = (vallidation_))

model.save('cats&dogs.h5')

loaded_model = keras.models.load_model('/content/cats&dogs.h5')
