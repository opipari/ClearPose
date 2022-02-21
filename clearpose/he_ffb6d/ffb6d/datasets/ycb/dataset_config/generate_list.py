import glob
import os
import numpy as np
import sys

assert(len(sys.argv) == 5)

dataset_list = sys.argv[1]
sensor = sys.argv[2]
template = sys.argv[3]
ratio = sys.argv[4] 

print(os.path.join(os.path.dirname(__file__), f"{dataset_list}_data_list.txt"))

f = open(os.path.join(os.path.dirname(__file__), f"{dataset_list}_data_list.txt"), "w+")
model_path = "/home/huijie/research/progresslabeller/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/data"

template_path = os.path.join(sensor, template)
# template_path = template


def all_select(f):
   for data_package in sorted(glob.glob(os.path.join(model_path, template_path))):
      for filepath in sorted(glob.glob(os.path.join(data_package, "*-box.txt"))):
         idx = os.path.basename(filepath)[:6]
         f.write(os.path.join(f"data/{sensor}/", os.path.basename(data_package), idx) + "\n")
         # f.write(os.path.join(f"data/", os.path.basename(data_package), idx) + "\n")
   f.close()

def random_select(f, ratio):
   for data_package in sorted(glob.glob(os.path.join(model_path, template_path))):
      files = sorted(glob.glob(os.path.join(data_package, "*-box.txt")))
      i = 0
      while i < len(files):
         filepath = files[i]
         idx = os.path.basename(filepath)[:6]
         f.write(os.path.join("data", sensor ,os.path.basename(data_package), idx) + "\n")
         # f.write(os.path.join("data", os.path.basename(data_package), idx) + "\n")
         i += int(1/ratio)
   f.close()   

if __name__ == "__main__":
   random_select(f, float(ratio))
   #all_select(f)
