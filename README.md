# MobileNet_V2
Implementation of Mobilenet V2 for binary image classification of dogs and cats using Keras and TensorFlow. 📚 Trained on a dataset of dogs and cats images, with customizable scripts for training, testing, and prediction on new data. 📊🛠️

# Dogs and Cats Classification with Mobilenet V2

Summary: This code implements a binary image classification model for dogs 🐶 and cats 🐱 using the Mobilenet V2 architecture in TensorFlow and Keras. The "cats_vs_dogs" dataset from TensorFlow Datasets is used to train, validate, and test the model.
The code applies data augmentation techniques 🧪 to improve the robustness of the model. Transfer learning 📜 is used to leverage the pre-trained weights of the Mobilenet V2 model, which significantly reduces the training time and increases the accuracy of the model. The trained model is saved in a .h5 file for future use.
The code also includes a script for making predictions on new images using the trained model.

To run the code, you will need to have TensorFlow, Keras, and TensorFlow Datasets installed. You can easily customize the code to work with your own dataset by modifying the data loading and preprocessing steps. The code is well-commented 💬 and should be easy to understand even if you are new to TensorFlow and Keras.
