# import os
# for dirname in os.listdir("."):
#     if os.path.isdir(dirname):
#         for i, filename in enumerate(os.listdir(dirname)):
#             os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")

import os
cwd = os.getcwd()
#dir2 = cwd +"/Real_dataset_1/"
dir1 = cwd +"/textures/markings/"
files = os.listdir(dir1)
files = sorted(files)
print(len(files))

def num_of_img_files_in_folder(Dir):
    list = os.listdir(Dir) # dir is your directory path
    number_files = len(list)
    return number_files

N = num_of_img_files_in_folder(dir1)
print(N)


n = 1
for i, filename in enumerate(files):
     #print(dir1 + filename +"  |  "+ dir2 + '{:06d}'.format(n) + ".png")
     #print(dir2 + filename[0:11] + ".jpg"+"  |  " + '{:06d}'.format(n) + ".jpg")
     
     os.rename(dir1 + filename, dir1 + '{:d}'.format(n) + ".jpg")
     #os.rename(dir2 + filename[0:11] + ".jpg", dir2 + '{:06d}'.format(n) + ".jpg")
     n += 1
