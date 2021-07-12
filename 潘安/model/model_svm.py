import os
import time


#load_data
def load_data(path):
    X = []
    y = []
    i=0
    for dirpath,dirsname,filesname in os.walk(path,topdown=True):
        for filename in filesname:
            filename_path = os.sep.join([dirpath,filename])
            #print(filename_path,i)
            X.append(filename_path)
            y.append(i-1)
        i+=1
    return X,y

X,y = load_data("./DataSet/cnn_char_train")



