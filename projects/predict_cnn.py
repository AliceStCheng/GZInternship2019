from keras.models import load_model # to load an externally trained model
from keras.preprocessing import image # for loading and working with images
from keras import optimizers # so that the SGD optimizer works
import numpy as np # for loading images into a list
import os # for reading through files in a directory into a list
import pandas as pd
import csv

# for plotting:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for roc curves and auc score:
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# image folder
folder_path = '/Users/lancastro/Desktop/Alice/cnn_data/val'
# path to model
model_path = '/Users/lancastro/Desktop/Alice/projects/classification_model.h5'
# dimensions of our images
img_width, img_height = 150, 150

opt = optimizers.SGD(lr=0.0001, momentum=0.9)
# load the model we saved
model = load_model('classification_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# summarize model.
model.summary()

# load all images into a list
images = []
image_names = []
i = 0
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    image_names.append(img)
    img = image.load_img(img, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
    # prints the number of loops it has done, which is the same as the
    # number of images it looked into to give a sense progress.
    i += 1
    print(i)

print("loops done!")


# stack up images list to pass for prediction
images = np.vstack(images)
class_indices = model.predict_classes(images, batch_size=10)
pred_pcnt = model.predict(images, batch_size=10)
print("prediction percentages:")
print(pred_pcnt)
print("finished predicting!")

# if you are looking for how to save the outputs into the same list
# I have to disappoint you because I couldn't and used Excel instead.
# But as separate files? That I can do. <= note: Brooke has thankfully
# solved the issue for me. The file for this is assign_labels.py.

# saves the image names as paths
df1 = pd.DataFrame(image_names, columns=['file_path'])
df1.to_csv('image_paths.csv', index=False)

# saves the predicted labels
indices = np.array(class_indices)
df2 = pd.DataFrame(indices)
df2.to_csv('predicted_classes.csv', index=False)

# saves the probability of each prediction
pcnt = np.array(pred_pcnt)
df3 = pd.DataFrame(pcnt)
df3.to_csv('probability.csv',
            line_terminator = ',\n',
            index=False)
print(df3)

# Getting the true labels for the ROC curve
colnames = ['name','label']
data = pd.read_csv('image_true_labels.csv', names=colnames)
labels = data.label.tolist()
# print(labels)

# probs = pd.read_csv('probability.csv', header=0)
# list_probs = probs.values.tolist()
# print(list_probs)

pred_test = np.array(pred_pcnt).T[0]

pred_test = np.array(pred_pcnt).T[0]

pred_img_name = [q.split("/")[-1] for q in image_names]
pred_df['image_filename'] = pred_img_name

es_folder     = '/Users/lancastro/Desktop/Alice/cnn_data/validation/extended_sources'
non_es_folder = '/Users/lancastro/Desktop/Alice/cnn_data/validation/non_es'
extended_list = glob.glob('%s/*.jpg' % es_folder)
non_es_list   = glob.glob('%s/*.jpg' % non_es_folder)
extended_class = np.zeros(len(extended_list)).astype(int)
non_es_class   = np.ones( len(non_es_list)).astype(int)


extended_file = [q.split("/")[-1] for q in extended_list]
non_es_file   = [q.split("/")[-1] for q in non_es_list]

# print(extended_list)
# print(extended_file)

# The following was where the bug came from:
# all_file_list = extended_list.copy()
# all_file_list.extend(non_es_list)

all_file_list = extended_file.copy()
all_file_list.extend(non_es_file)

# this isn't actually list so this is bad variable naming on Brooke's part
# it's an array. they behave differently. maybe rename or it will be confusing later
all_class_list = np.append(extended_class, non_es_class)

# print(len(extended_class), len(non_es_class), all_class_list.shape, len(all_file_list))

true_classes = pd.DataFrame(all_file_list)
true_classes.columns = ['file_name']
true_classes['class'] = all_class_list

matched_true_pred = pd.merge(true_classes, pred_df, left_on="file_name",
                             right_on="image_filename",
                             how="inner",
                             suffixes=("_true", "_pred"))

print(len(true_classes), len(pred_df))
print(len(matched_true_pred))

print(true_classes)
print(pred_df['image_filename'])

print(matched_true_pred.head())
matched_true_pred.to_csv('matched_true_pred_classes.csv')

# for plotting:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for roc curves and auc score:
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


y_true = np.array(matched_true_pred['class'])
y_pred = np.array(matched_true_pred['prediction_percentage'])

fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

# function for plotting the ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.show()



plot_roc_curve(fpr, tpr)
auc = roc_auc_score(y_true, y_pred)
print("The AUC is: ")
print(auc)

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)


average_precision = average_precision_score(y_true, y_pred,
                                                     average="micro")

# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('pr_curve.png')
plt.show()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# code predicting a single image - should you want it.

# img = image.load_img(
#         '/Users/lancastro/Desktop/Alice/cnn_data/validation/extended_sources/GDS_13826.jpg',
#         target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
#
# images = np.vstack([x])
# classes = model.predict_classes(images, batch_size=10)
# print(classes)
