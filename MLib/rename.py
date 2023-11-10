import os
import shutil
import time

def rename(input_path, output_path):
    files = os.listdir(input_path)
    i = 1
    n = 1
    while i < len(files)+1:
        try:
            #print('{:06d}'.format(i) + ".jpg")
            shutil.copyfile(input_path + '{:d}'.format(n) + ".png", output_path + '{:06d}'.format(i) + ".jpg")
            #os.rename(input_path + '{:d}'.format(n) + ".png", output_path + '{:06d}'.format(i) + ".jpg")
            i += 1
        except:
            pass
        n += 1