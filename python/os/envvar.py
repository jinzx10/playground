import os

cpath = os.environ['CPATH']
cpath = ":" + cpath
cpath = cpath.replace(":"," -isystem")

print(cpath)
