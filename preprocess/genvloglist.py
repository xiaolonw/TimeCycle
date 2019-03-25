
import numpy as np
import os

src = 'YOUR_DATASET_FOLDER/manifest.txt'
outlist = 'YOUR_DATASET_FOLDER/vlog_frames_12fps.txt'
foldername = 'YOUR_DATASET_FOLDER/vlog_frames_12fps/'

file = open(src, 'r')
fout = open(outlist, 'w')

for line in file:
    line = line[:-1]
    fname = foldername + line
    fnms  = len(os.listdir(fname))

    outstr = fname + ' ' + str(fnms) + '\n'
    fout.write(outstr)


file.close()
fout.close()
