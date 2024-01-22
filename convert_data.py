import sys;
import os;
import numpy as np ;
from PIL import Image;

def balanceData(input_data, input_labels, numSamples):
    labels = np.unique(input_labels)
    num_labels = len(labels)
    sample_indices = [];

    # list for out output data
    output_data = []
    output_labels = []
    for label in labels:
        indices = np.where(input_labels == label)[0]
        sample_indices.extend(np.random.choice(indices, size=numSamples))

        #build labels in one hot foramt
        label_OneHot = np.zeros(num_labels)
        label_OneHot[label] = 1
        labels_samples = [label_OneHot] * numSamples

        output_labels.extend(labels_samples)

    output_data = input_data[sample_indices];

    random_order = np.arange(0, num_labels * numSamples)
    np.random.shuffle(random_order)
    # and applying it on our outputs (fancy indexing)
    output_data_r = output_data[random_order]
    output_labels = np.array(output_labels)
    output_labels_r = output_labels[random_order]

    return output_data_r, output_labels_r
    
if __name__ == "__main__":

    #Check sys.args
    if len(sys.argv) != 3:
        print("Usage: convert_data.py <path_to_images> <filename.npz>");
        exit(1);
        
    imgPath = sys.argv[1];
    npzFileName = sys.argv[2];

    #Data analysis and preprocessing

    if(os.path.exists(imgPath) == False):
       exit("Image Path does not exist");

    # Get a list of image file names in the folder
    file_names = os.listdir(imgPath);

    # Initialize an empty list to store images as arrays
    image_data = []
    image_labels = []
    numSamples = {};

    # Loop through the images and convert them to arrays
    for file_name in file_names:
        img = Image.open(imgPath + "/" + file_name)
        img_array = np.array(img)

        #scale images from [0, 255] to [0,1]
        img_array = img_array / img_array.max();
        
        #extract the labels
        image_label = int(file_name.split("-")[0])

        # Build dictionary
        if image_label in numSamples:
            numSamples[image_label] += 1
        else:
            numSamples[image_label] = 1

        #build list
        image_data.append(img_array)
        image_labels.append(image_label);

    image_data = np.array(image_data)
    image_labels = np.array(image_labels);

    balanced_data, balanced_labels = balanceData(image_data, image_labels, 6000);

    # Save the numpy array as an npz 
    np.savez(npzFileName, balanced_data, balanced_labels)
    # np.savez(npzFileName, image_data[indices], image_labels[indices])

