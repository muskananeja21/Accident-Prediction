import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2

# Define constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Define data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
train_generator = train_datagen.flow_from_directory('D:/College/Kodeshetra/train', target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory('D:/College/Kodeshetra/val', target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)), #For a given window of ksize , takes the maximum value within that window. Used for reducing computation and preventing overfitting.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
)


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=val_generator)

# Evaluate the model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('D:/College/Kodeshetra/test', target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='binary')
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Defining a function to run prediction through an image array
def predict_frame_class(frame, model):
    # Convert the frame to a tensor
    tensor = tf.keras.preprocessing.image.img_to_array(frame)
    tensor = tf.expand_dims(tensor, axis=0)

    # Preprocess the tensor
    tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)

    # Make a prediction
    prediction = model.predict(tensor)

    # Return the predicted class (0 or 1) 
    return int(prediction[0][0])


# Read in the video file
video_path = "D:/TestV.mp4"
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the desired frame size
frame_size = (224, 224)

# Loop over every 20th frame
for i in range(0, total_frames, 20):
    # Set the current frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    # Read the frame
    ret, frame = cap.read()

    # If the frame was successfully read
    if ret:
        # Resize the frame to the desired size
        frame = cv2.resize(frame, frame_size)

        # Run the frame through the model to get the predicted class
        predicted_class = predict_frame_class(frame, model)

        # Do something with the predicted class (e.g., print it out)
        print(predicted_class)

# Release the video capture object
cap.release()
