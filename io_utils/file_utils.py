import os,glob

def ensure_dir(p):os.makedirs(p,exist_ok=True)

def list_images(f,exts=('.png','.jpg','.jpeg','.bmp')):
 l=[]
 [l.extend(glob.glob(os.path.join(f,'*'+e))) for e in exts]
 return sorted(l)
