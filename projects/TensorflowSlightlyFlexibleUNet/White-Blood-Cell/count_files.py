import os
import traceback


def count_files(dir):
  count = 0
  if os.path.exists(dir):
     count = sum(len(files) for _, _, files in os.walk(dir))
  else:
    print("---Not found dir {}".format(dir))

  return count
  
if __name__ == "__main__":
  try:
    root_dir = "../../../dataset/Cervical-Cancer/"

    subdirs = ["test/Metaplastic/images/",
               "train/Metaplastic/images/",
               "valid/Metaplastic/images/",
               ]
    print("Metaplastic, ")
    print("dataset, count") 
    for subdir in subdirs:
       count  = count_files(root_dir + subdir)
       dir = subdir.replace("Metaplastic/", "")
       print("{},{}".format(dir, count))

 
  

  except:
    traceback.print_exc()
