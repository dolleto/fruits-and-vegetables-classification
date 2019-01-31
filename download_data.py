import os
import shutil
from subprocess import call

url = 'https://www.dropbox.com/s/oejyy6w4lzr071v/data.zip?dl=0'
cmd = 'wget {} --no-check-certificate'.format(url)
call(cmd, shell=True)

call('unzip data.zip?dl=0', shell=True)
