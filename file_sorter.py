import os, shutil
import pandas as pd
import sys
import glob
from pathlib import Path


# folder you want to move from
source = '/Users/lancastro/Desktop/Alice/gz_candels_subjects'
# in this case, dest 1 is stars, dest 2 is extended sources, where dest is destination.
dest1 = '/Users/lancastro/Desktop/Alice/bright_stars'
dest2 = '/Users/lancastro/Desktop/Alice/extended_sources'
csv_file = '/Users/lancastro/Desktop/Alice/projects/bright_star_list.csv'


def list__column_files(f):
    #location of the list of images
    imgs_in = pd.read_csv(f)
    #making a list of filenames from the source file with a for loop
    #hubble_filename = ['%s%s.jpg' % ('', q) for q in imgs_in['hubble_id_img']]
    imgs_in = df.add_suffix('hubble_id_img')
    return(hubble_filename)

print(list__column_files(csv_file))

def list_dir_files(path):
    list = []
    for filename in Path(path).glob('**/*.jpg'):
        list.append(filename)
    [s.replace('(', ')') for s in list]
    print(list)

def move_files(f):

    img_list = list_files(f)
    files = os.listdir(source)

    # os.walk(source) returns a list of names of every file in the directory.
    # the following checks whether the names in the list from the .csv file
    # match any of that in the main directory.
    # if any(names in list_files(f) for names in os.walk(source)):
    #     print('yay! =D')
    # else:
    #     print('nay. =C')
    for file in range(len(img_list)):
        if img_list[file] == os.walk(source)[file]:
            print("yay! =D")
        else:
            print("nay. =C")
        file += 1
#print(list_dir_files(source))
#print(type(list_files(csv_file)[5]))
#print(type(os.walk(source)[11]))
#print(any(elem in list_files(csv_file) for elem in os.walk(source)))
#move_files(csv_file)
#print(list_files(csv_file), file=open("bright_stars.txt", "a"))
#print(os.listdir(source), file=open("output.txt", "a"))
