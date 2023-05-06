loaded_model = keras.models.load_model('/content/cats&dogs.h5')

loaded_model.evaluate(test_)

plt.figure(figsize = (4,4))

for image , _ in test_.take(90) : 
    pass

pre = loaded_model.predict(image)

plt.figure(figsize = (10 , 10))
j = None
for value in enumerate(pre) : 
    plt.subplot(7,7,value[0]+1)
    plt.imshow(image[value[0]])
    plt.xticks([])
    plt.yticks([])
    if value[1] > pre.mean() :
        j = 1
        color = 'blue' if j == _[value[0]] else 'red'
        plt.title('dog' , color = color)
    else : 
        j = 0
        color = 'blue' if j == _[value[0]] else 'red'
        plt.title('cat' , color = color)

plt.show()
