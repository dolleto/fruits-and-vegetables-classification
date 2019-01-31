import os
import shutil
from subprocess import call

url = # à compléter
cmd = 'wget {} --no-check-certificate'.format(url)
call(cmd, shell=True)

call('unzip data.zip', shell=True)
