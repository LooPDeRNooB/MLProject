import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras import layers, models

def load_data(npz_path):
    # Load data and labels from the npz file
    with np.load(npz_path) as data:
        images, labels = data["arr_0"], data["arr_1"]
    return images, labels

def create_simple_dnn(input_shape, num_classes):

    activation_function = "relu"

    model = tf.keras.Sequential() ;
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(128, activation=activation_function));
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"));

    #Why is the performance of the model so inconsistent?
    # model = tf.keras.Sequential() ;
    # model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    # model.add(tf.keras.layers.Dense(128, activation="relu"));
    # model.add(tf.keras.layers.Dense(num_classes, activation="relu"));
    # model.add(tf.keras.layers.Softmax());

    return model;


def create_lenet5_cnn(input_shape, num_classes):
    # https://www.datasciencecentral.com/lenet-5-a-classic-cnn-architecture/

    activation_function = "relu"

    #First layer: image which passes through the first convolutional layer with 6 feature maps or filters having size 5×5 and a stride of one
    model = tf.keras.Sequential();
    model.add(tf.keras.layers.Conv2D(6, (5, 5), (1, 1), input_shape=input_shape, activation=activation_function));

    #Second layer: Average Pooling layer or sub-sampling layer with a filter size of 2×2 and a stride of 2
    model.add(tf.keras.layers.AveragePooling2D((2, 2), (2, 2)));

    #Third layer: second convolutional layer with 16 feature maps having size 5×5 and a stride of 1
    model.add(tf.keras.layers.Conv2D(16, (5, 5), (1, 1), activation=activation_function));

    #Fourth layer: Average Pooling layer or sub-sampling layer with a filter size of 2×2 and a stride of 2
    model.add(tf.keras.layers.AveragePooling2D((2, 2), (2, 2)));

    #Fifth layer: fully connected convolutional layer with 120 feature maps each of size 1×1
    # das hier vorher?
    # model.add(tf.keras.layers.Conv2D(120, (5, 5), (1, 1), activation=activation_function))
    model.add(tf.keras.layers.Flatten());
    # model.add(tf.keras.layers.Dense(120, activation=activation_function));

    #Sixth layer: fully connected layer with 84 units
    model.add(tf.keras.layers.Dense(84, activation=activation_function));

    #Seventh layer: output layer with num_classes units
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"));

    return model;


def train_test_split(images, labels, train_size=0.8):
    # Split data into training and testing sets
    randomIndices = np.arange(0, images.shape[0])
    np.random.shuffle(randomIndices)
    trainInd, testInd = np.split(randomIndices, [int(train_size * randomIndices.size)])
    X_train = images[trainInd]
    Y_train = labels[trainInd]
    X_test = images[testInd]
    Y_test = labels[testInd]

    return X_train, X_test, Y_train, Y_test

def train_model(model, X_train, y_train, epochs=10, batch_size=64):
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)
    accuracy = accuracy_score(Y_test_classes, Y_pred_classes)
    return accuracy


def show_data(X_test, Y_test, model, model_name="Unknown Model"):

    Y_pred = model.predict(X_test);

    Y_pred_scalar = np.argmax(Y_pred, axis=1) ;
    Y_test_scalar = Y_test.argmax(axis=1) ;

    cm_helper = np.zeros([Y_test.shape[1],Y_test.shape[1],X_test.shape[0]]) ;
    cm_helper[Y_test_scalar,Y_pred_scalar,range(0,X_test.shape[0])] = 1 ;
    
    # reduce away the artificial axis from the helper matrix to obtain the CM
    cm = cm_helper.sum(axis=2) ;
    
    plt.title(f"Confusion Matrix on Test Data - {model_name}")
    plt.xlabel("Prediction")
    plt.ylabel("True Labels")

    # display
    plt.imshow(cm) ;
    plt.show()



if __name__ == "__main__":
    
    if len(sys.argv) != 6:
        print("Usage: cnn.py <npz> imgWidth imgHeight imgChannels <mode>");
        exit(1);

    # Parse command line arguments
    npz_path = sys.argv[1]
    img_width = int(sys.argv[2])
    img_height = int(sys.argv[3])
    img_channels = int(sys.argv[4])
    mode = sys.argv[5]

    # Load data and labels
    images, labels = load_data(npz_path)

    images = images.reshape(images.shape[0], img_height, img_width, img_channels)

    # Define number of classes based on labels, since they are one-hot encoded
    num_classes = labels.shape[1]

    #Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, train_size=0.8)

    if mode == "train":
        epochs = 10;
        batch_size = 128;

        # Create and train simple dnn
        print("Training Simple DNN...")

        dnn_model = create_simple_dnn((img_height, img_width, img_channels), num_classes)
        train_model(dnn_model, X_train, Y_train, epochs=epochs, batch_size=batch_size)
        dnn_model.save("./models/dnn_model.keras")

        # Create and train LeNet-5 CNN
        print("Training LeNet-5 CNN...")
        
        cnn_model = create_lenet5_cnn((img_height, img_width, img_channels), num_classes)
        train_model(cnn_model, X_train, Y_train, epochs=epochs, batch_size=batch_size)
        cnn_model.save("./models/cnn_model.keras")

    elif mode == "test":
        # Load pre-trained model
        dnn_model = models.load_model("./models/dnn_model.keras")
        cnn_model = models.load_model("./models/cnn_model.keras")

        # Evaluate models on test data
        dnn_accuracy = evaluate_model(dnn_model, X_test, Y_test)
        cnn_accuracy = evaluate_model(cnn_model, X_test, Y_test)

        print(f"Simple DNN Accuracy: {dnn_accuracy:.4f}")
        print(f"LeNet-5 CNN Accuracy: {cnn_accuracy:.4f}")

        show_data(X_test, Y_test, dnn_model, "Simple DNN")
        show_data(X_test, Y_test, cnn_model, "LeNet-5 CNN")