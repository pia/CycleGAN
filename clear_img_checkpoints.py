"""
删掉imgs和checkpoints目录下的文件
"""
import os

dirs = ['imgs/', 'checkpoints/']

for d in dirs:
    files = os.listdir(d)
    for f in files:
        to_del = d+f
        os.remove(to_del)
        print('Del ' + to_del)
