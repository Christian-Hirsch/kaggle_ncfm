import os
import subprocess

ROOT = "/home/ubuntu/kagglencfm/data"
PROCESSED = "{}/processed".format(ROOT)
RAW = "{}/raw".format(ROOT)


def unzip(src):
 subpro("unzip {}/{}.zip -d {}/".format(RAW, src, PROCESSED))

def mkdir():
 subpro("mkdir {}/alldata".format(PROCESSED))

def mv_gen(src):
 p1 =  subpro("find {}/{} -type f -print0".format(PROCESSED, src), stdout = subprocess.PIPE)
 p2 = subpro("xargs -0 mv -t {}/alldata/".format(PROCESSED), stdin= p1.stdout)

def ln():
 subpro("ln -s {}/alldata -t ./data/".format(PROCESSED))

def wget():
 subpro("wget http://www.platform.ai/models/vgg16_bn_conv.h5 -P ./models/".format(PROCESSED))
 

def subpro(command, **kwargs):
 subprocess.Popen(command.split(" "), **kwargs)

[unzip(folder) for folder in ['train', 'test_stg1']]

mkdir()

fish = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
fish = ['train/{}'.format(typ) for typ in fish]
fish += ['test_stg1']

[mv_gen(subfolder) for subfolder in fish]

ln()
wget()
