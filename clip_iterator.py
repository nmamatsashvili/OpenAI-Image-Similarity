from importlib.resources import path
import os
from pickle import FALSE
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from datetime import datetime

model = SentenceTransformer('clip-ViT-B-32')

stmtIds = []
stmtIdsUnvisited = []
print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
dir = "C:/Users/nmamatsashvili/source/repos/WebScraping/ParserApp/bin/Debug/net6.0/responses_test_50"
p = Path(dir)
i = 0

f = open(f"data.txt", "x")
f.close()


for folder in os.listdir(dir):
    stmtIds.append(folder)
    i = i + 1

stmtIdsUnvisited = stmtIds

for stmt in stmtIds:
    stmtIdsUnvisited.remove(stmt)
    for image in os.listdir(dir + "/" + stmt):
        if os.fsdecode(image).endswith(".jpg") == False:
            continue
        img_emb = model.encode(Image.open(f"{dir}/{stmt}/{image}"))
        #img_emb = str(stmt) + "-" + image
        #print("current image: " + img_emb)
        for stmtNext in stmtIdsUnvisited:
            for imageNext in os.listdir(dir + "/" + stmtNext):
                if os.fsdecode(imageNext).endswith(".jpg") == False:
                    continue
                img_emb_next = model.encode(Image.open(f"{dir}/{stmtNext}/{imageNext}"))
                #img_emb_next = str(stmtNext) + "-" + imageNext
                #print("next image: " + img_emb_next)
                cos_scores = util.cos_sim(img_emb, img_emb_next)
                result = f"statements: {stmt} vs {stmtNext} - images: {image} vs {imageNext} - cos_score: {cos_scores}" 
                openFile = open(f"data.txt", "a")
                openFile.write("\n" + result)

openFile.close()
print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])





"""
for folder in os.listdir(dir):
    stmtIds.append(folder)
    i = i + 1
    if i > 2:
        continue
    for image in os.listdir(dir + "/" + folder):
             filename = os.fsdecode(image)
             if filename.endswith(".jpg") == False:
                 continue
             
             img_emb = model.encode(Image.open(f"{dir}/{folder}/{image}"))
             torch.save(img_emb, f"{dir}/{folder}/tensors.pt")
             #torch.load(f"{dir}/{folder}/tensors.pt") #https://discuss.pytorch.org/t/save-a-tensor-to-file/37136/4
    print(f"embedding for folder {folder} ended successfully")

"""