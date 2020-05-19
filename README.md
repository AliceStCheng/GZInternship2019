# GZ_cnn
My internship using a cnn to classify galactic morphology.

Note that the code will not work straight away as the data is stored locally.

24000 images were used to train the network. Images sourced from the Sloan Digital Sky Survey.

The network has achieved a classification accuracy of 78%, an AUC of 0.76, where 1 is where all the results have a true positive classification. Out of all the misclassified images, the network conservatively classified noisy images as non-galaxies, producing a reliable set of galaxy images which can be used for further classification.

For summary and results, see poster named 'internship_poster.pdf'.
