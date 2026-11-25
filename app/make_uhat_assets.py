"""Save one UHaT test image per character class + the 40 class names to app/assets/uhat_examples.json.
The app uses them for the VAE interpolation and the character dropdowns. Run once wherever UHaT
is unzipped (starter/data/uhat), from the repo root:   python app/make_uhat_assets.py"""
import os
import sys
import json

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(os.path.dirname(HERE),"starter"))
from uhat import load_uhat_characters

root=os.path.join(os.path.dirname(HERE),"starter","data","uhat")
_,_,names=load_uhat_characters(root=root,split="train")
x,y,_=load_uhat_characters(root=root,split="test",class_names=names)

images=[]
for c in range(len(names)):
    i=int((y==c).nonzero()[0])                     #first test image of each class
    images.append((x[i,0]*255).round().byte().flatten().tolist())

os.makedirs(os.path.join(HERE,"assets"),exist_ok=True)
with open(os.path.join(HERE,"assets","uhat_examples.json"),"w") as f:
    json.dump({"class_names":names,"images":images},f)
print("Saved",len(names),"classes:",names)
