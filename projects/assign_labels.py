from predict_cnn import image_names

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

# print(extended_file)

all_file_list = extended_list.copy()
all_file_list.extend(non_es_list)

# this isn't actually list so this is bad variable naming on Brooke's part
# it's an array. they behave differently. maybe rename or it will be confusing later
all_class_array = np.append(extended_class, non_es_class)

# print(len(extended_class), len(non_es_class), all_class_array.shape, len(all_file_list))

true_classes = pd.DataFrame(all_file_list)
true_classes.columns = ['file_name']
true_classes['class'] = all_class_array

matched_true_pred = pd.merge(true_classes, pred_df, left_on="file_name",
                             right_on="image_filename", how="inner",
                             suffixes=("_true", "_pred"))

print(len(true_classes), len(pred_df))
print(len(matched_true_pred))

print(matched_true_pred.head())
matched_true_pred.to_csv('matched_true_pred_classes.csv')
