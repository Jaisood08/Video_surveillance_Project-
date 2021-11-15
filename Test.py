from imutils import paths
import face_recognition
import pickle
import cv2
import os
import sys
import numpy
from numpy import append


File_name = 'Testclip.mp4'
OUT_name = "Output.mp4"
encodingsFileN = "New_Encoding.pickle"
encodingsFile = "encodings.pickle"

try:
    data = pickle.loads(open(encodingsFile, "rb").read())
except:
    data = {"encodings": [], "names": []}

VP = cv2.VideoCapture(File_name)
if not VP.isOpened():
    print("Cannot open File")
    exit()

# Video_Details_capture
fps = VP.get(cv2.CAP_PROP_FPS)
W = int(VP.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(VP.get(cv2.CAP_PROP_FRAME_HEIGHT))
Frame_size = ((W, H))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print("codec = ", fourcc, ", Fps = ", fps, " , Dimension = ", Frame_size)

# Converting_To_GreyScale_and_Save
success, image = VP.read()
count = 0
Ident = []
Ident_image = []
SC ={}
UK = 0

while success:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names =[]
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
    print(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        if(name == "Unknown"):
            encodings = numpy.array(encodings)
            try:
                encodings = encodings.reshape((128, ))
                name = "Unknown_"+str(UK)
                data["encodings"].append(encodings)
                data["names"].append("Unknown_"+str(UK))
                Ident.append("Unknown_"+str(UK))
                UK += 1
                img = image[top:bottom,left:right]
                Ident_image.append(img)
                SC[name]=0
            except:
                print("Kuch NA Mila")
        
        else:
            try:
                SC[name]+=1
            except:
                SC[name]=0
            if "Unknown" in name:
                encodings = numpy.array(encodings)
                try:
                    encodings = encodings.reshape((128, ))
                    data["encodings"].append(encodings)
                    data["names"].append(name)
                except:
                    print("Kuch NA Mila")
            if name not in Ident:
                SC[name]=0
                Ident.append(name)
                img = image[top:bottom,left:right]
                Ident_image.append(img)

    success, image = VP.read()
    count += 1

VP.release()
cv2.destroyAllWindows()
j = 0

Pim ={}

for i in Ident_image:
    img = i
    pad = numpy.full((30,img.shape[1],3), [255,255,255], dtype=numpy.uint8)
    result = numpy.vstack((img,pad))
    X ,Y = result.shape[0],result.shape[1]
    
    if "Unknown" in Ident[j]:
        val = Ident[j]
        for index, item in enumerate(data["names"]):
            if item ==  Ident[j]:
                data["names"][index] = val
        try:
            time = (1/fps)*(SC[Ident[j]]+1)
        except:
            time =  (1/fps)
        Ident[j] = val
        if Ident[j] not in Pim:
            L = []
            L.append(i)
            L.append(time)
            Pim[Ident[j]] = L
        else:
            Pim[Ident[j]][1] += time
    else:
        time = (1/fps)*(SC[Ident[j]]+1)
        if Ident[j] not in Pim:
            L = []
            L.append(i)
            L.append(time)
            Pim[Ident[j]] = L
        else:
            Pim[Ident[j]][1] += time
    j+=1
    


print("Total Frames = ", count)

print("Saving Encodings...")
f = open(encodingsFileN, "wb")
f.write(pickle.dumps(data))
f.close() 


print("Saving graph.")
f = open("GRAPH.pickle", "wb")
f.write(pickle.dumps(Pim))
f.close() 

os.system('python Map.py')

print("Done")

cv2.destroyAllWindows()