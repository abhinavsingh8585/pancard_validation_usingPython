

#pan card validation
import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt              
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout




img_size = 100
datadir = r'images'    # root data directiory
CATEGORIES = os.listdir(datadir)
print(CATEGORIES)




# Preprocessing Function
def PreProcess(img_size, path):
    """This function reads images from the given folders subfolder 
        and returns a normalized array along with their respective classes"""
    x, y = [], []
    CATEGORIES = os.listdir(path)
    print("Found {} classes: {}".format(len(CATEGORIES), CATEGORIES))
    
    for category in CATEGORIES:
        path = os.path.join(datadir, category)
        classIndex = CATEGORIES.index(category)
        
        for imgs in tqdm(os.listdir(path)):
            img_arr = cv2.imread(os.path.join(path, imgs))

            # resize the image
            resized_array = cv2.resize(img_arr, (img_size, img_size))
            cv2.imshow("images", resized_array)
            cv2.waitKey(1)
            # Normalize the image 
            resized_array = resized_array/255.0
            x.append(resized_array)
            y.append(classIndex)
    cv2.destroyAllWindows()
    return x, y, CATEGORIES

x, y, CATEGORIES = PreProcess(img_size, datadir)





# Split the dataset into training and testing
X_train, x_test, Y_train, y_test = train_test_split(x, y, random_state=42)

# Convert all the list to numpy array
X_train = np.array(X_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
y_test = np.array(y_test)





# Build the model
model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(img_size, img_size, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), strides=2, activation="relu"))
model.add(Conv2D(64, (3, 3),  activation="relu"))
model.add(Conv2D(8, (3, 3), strides=2, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()




# Train the model
history = model.fit(X_train, Y_train, batch_size = 2, epochs=15, verbose=1)


accuracy = history.history['accuracy']
loss = history.history['loss']


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(accuracy)
ax1.set_title("Accuracy")
ax2.plot(loss)
ax2.set_title('Loss')


# evaluate the model
model.evaluate(x_test, y_test)

print("1")

# Load images
img_real = cv2.cvtColor(cv2.imread('cards/real.jpg'), cv2.COLOR_BGR2RGB)
img_fake = cv2.cvtColor(cv2.imread('cards/fake.jpg'), cv2.COLOR_BGR2RGB)
real = np.expand_dims(cv2.resize(img_real, (img_size, img_size)), axis=0)/255.0
fake = np.expand_dims(cv2.resize(img_fake, (img_size, img_size)), axis=0)/255.0





fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img_real)
ax1.set_title("Real")
ax2.imshow(img_fake)
ax2.set_title('Fake')





# Predict from a image
pred1 = model.predict(real)
pred2 = model.predict(fake)
print(CATEGORIES[np.argmax(pred1)], CATEGORIES[np.argmax(pred2)])





# Export the model
model.save("validation.h5")

