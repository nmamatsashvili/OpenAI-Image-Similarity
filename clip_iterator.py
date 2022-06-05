import os
import time
from importlib.resources import path
from pickle import FALSE
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from pathlib import Path


model = SentenceTransformer('clip-ViT-B-32')

stmtIds = []
stmtIdsUnvisited = []
start = time.perf_counter()

dir = "C:/Users/nmamatsashvili/source/repos/WebScraping/ParserApp/bin/Debug/net6.0/responses_test_50"
dataFile = "data50.txt"
FileExists = os.path.exists(dataFile)
if FileExists :
    file = open(dataFile,"r+")
    file.truncate()
    file.close()


for folder in os.listdir(dir):
    stmtIds.append(folder)

stmtIdsUnvisited = list(stmtIds)
openFile = open(dataFile, "a")

for stmt in stmtIds:
    stmtIdsUnvisited.remove(stmt)
    lstImagesCurrent = []
    
    for image in os.listdir(dir + "/" + stmt):
        if os.fsdecode(image).endswith(".jpg") == False:
            continue
        lstImagesCurrent.append(Image.open(f"{dir}/{stmt}/{image}"))

    imgEmbeddingsCurrentBatch = model.encode(lstImagesCurrent)
    indx = 0
    for img_emb in imgEmbeddingsCurrentBatch:
        
        for stmtNext in stmtIdsUnvisited:
            lstImagesNext = []
            for imageNext in os.listdir(dir + "/" + stmtNext):
                if os.fsdecode(imageNext).endswith(".jpg") == False:
                    continue
                lstImagesNext.append(Image.open(f"{dir}/{stmtNext}/{imageNext}")) 
            imgEmbeddingsNextBatch = model.encode(lstImagesNext) 
            indxNext = 0
            for embNext in imgEmbeddingsNextBatch:
                cos_scores = util.cos_sim(img_emb, embNext)
                img = os.listdir(dir + "/" + stmt)[indx]
                imgNxt = os.listdir(dir + "/" + stmtNext)[indxNext]
                result = f"statements: {stmt} vs {stmtNext} - images: {img} vs {imgNxt} - cos_score: {cos_scores}" 
                openFile.write("\n" + result)
                indxNext += 1
        indx += 1

openFile.close()
end = time.perf_counter()
print("total time: " + str( round(end - start), 2) + " seconds")
