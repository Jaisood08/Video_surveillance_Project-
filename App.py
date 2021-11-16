import os
from flask import Flask, flash, request, redirect, url_for,render_template,send_from_directory
from werkzeug.utils import secure_filename
import pickle
from imutils import paths
import face_recognition
from zipfile import ZipFile
import cv2
import os
import sys
import numpy
from numpy import append
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4','m4a','m4v','f4v','f4a','m4b','m4r','f4b','mov','3gp','3gp2','3g2','3gpp','3gpp2','ogg','oga','ogv','ogx','wmv','wma','asf','webm','flv','avi','qt'}
Allowed_Encoding = {'pickle'}
Allowed_Dataset = {'zip'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "abc"

def Hole_Test(video,encoding):
    File_name = "uploads/"+video
    encodingsFile = "uploads/"+encoding
    encodingsFileN = "uploads/New_"+encoding

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

    data = Pim
    from io import BytesIO
    colors = ['crimson', 'dodgerblue', 'teal', 'limegreen', 'gold','blue','green','red','cyan','magenta','yellow','black']
    values = []
    for key in data:
        print(key)
        values.append(data[key][1])

    labels = list(data.keys())
    colors = colors[0:len(labels)]

    print(values)
    height = 0.9

    plt.barh(y=labels, width=values, height=height, color=colors, align='center')

    i = 0
    for key in data:
        value = data[key][1]
        content = "temp.png"
        cv2.imwrite(content,data[key][0])
        im = mpimg.imread(content)
        print(type(im))
        print(im.shape)
        plt.imshow(data[key][0], extent=[value - 0.01, value - 0.05, i - height + 0.5 , i + height -0.5], aspect='auto', zorder=2)
        extent=[value - 8, value - 2, i - height / 2, i + height / 2]
        print(extent)
        i+=1

    plt.xlim(0, max(values) * 1.05)
    plt.ylim(-0.5, len(labels) - 0.5)
    plt.tight_layout()
    plt.xlabel('Seconds')
    plt.ylabel('Persons')

    mapsave = os.path.splitext(File_name)[0]+".png"
    plt.savefig(mapsave, facecolor='w', bbox_inches="tight",
                pad_inches=0.3, transparent=True)
    print("Saved")
    return (os.path.splitext(video)[0]+".png")

def Encoding_Make(file):
    wd = os.getcwd()
    print("Working in == ",wd)
    file_name = "uploads/"+file
    Out_name = file[:-4]
    extract = file_name[:-4]

    with ZipFile(file_name, 'r') as zip:
        zip.printdir()
        print('Extracting all the files now...')
        zip.extractall(path=extract)
        print('Done!')

    #Essential
    DataFolder = extract+"/"+next(os.walk(extract))[1][0]
    encodingsFile = extract+".pickle"
    detection_method = 'cnn'

    print("Starting Encoding")
    imagePaths = list(paths.list_images(DataFolder))
    knownEncodings = []
    knownNames = []
    name = ' '
    F = 1
    F_C = 0

    #Calculating Encoding
    for (i, imagePath) in enumerate(imagePaths):    
        name1 = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (name1!=name):
            name = name1
            list1 = os.listdir(DataFolder+"/"+name)
            F_C = len(list1)
            F = 1
        print(name," ",F,"/",F_C)
        F+=1
        try:
            boxes = face_recognition.face_locations(rgb,model=detection_method)
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
        except:
            print("failed")        
    #Saving Encodings
    print("Saving Encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(encodingsFile, "wb")
    f.write(pickle.dumps(data))
    f.close()        

    print("Done -> ",encodingsFile)
    os.chdir(wd)  


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file1(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Allowed_Encoding

def allowed_file2(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Allowed_Dataset

@app.route('/', methods = ['GET','POST'])
def Connect():
    # return render_template("home.html",filename = "Testclip.png")
    return render_template("home.html")

@app.route('/train', methods = ['GET','POST'])
def Connect2():
    return render_template("Train.html")

@app.route('/dataset', methods = ['GET','POST'])
def datasetup():
    if request.method == 'POST':
      file = request.files['file']
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      if not allowed_file2(filename):
          flash('Extension Not Allowed')
          return render_template("test.html")
      
      encodingsFile = "uploads/"+filename[:-4]+".pickle"
      print(encodingsFile)
      print(filename,"File Saved")
      Encoding_Make(filename)
      print("aya bhi tha yha pe")
      try:
          filename = filename[:-4]+".pickle"
          print(encodingsFile)
          return render_template("Train.html",filename=filename)
          print("aya bhi tha yha pe")
      except Exception as e:
          return str(e)
    else:
        return render_template("Train.html")

@app.route('/uploads/<filename>',methods=['GET'])
def uploads(filename):
    return send_from_directory('uploads', filename)

@app.route('/Video', methods = ['GET','POST'])
def video():
    if request.method == 'POST':
        try:
            vfile = request.files['file']
            vfilename = secure_filename(vfile.filename)
            print(vfilename)
            if not (vfilename == ""):
                if not allowed_file(vfilename):
                    flash('Extension Not Allowed')
                    return render_template("home.html")
                else:
                    flash('Video Upload Success')
                vfile.save(os.path.join(app.config['UPLOAD_FOLDER'], vfilename))      
                print(vfilename,"  File Saved")
            else:
                flash('Invalid')
                return render_template("home.html")
        except:
            flash('Invalid')
        
        try:
            enfile = request.files['vfile']
            filename = secure_filename(enfile.filename)
            print("filename == ",filename)
            if not (filename == ""):
                if not allowed_file1(filename):
                    flash('Extension Not Allowed')
                    return render_template("home.html")
                else:
                    flash('Encoding Uploaded Successfully')
                enfile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))      
                print(filename,"  File Saved")
        except:
            flash('Invalid')
            print("No Encoding Recived")

        name = Hole_Test(vfilename,filename)
        
        return render_template("home.html",filename =  name)
    else:
        return render_template("Train.html")

if __name__=="__main__":
	app.run(debug = True)