import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
import pickle

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4','m4a','m4v','f4v','f4a','m4b','m4r','f4b','mov','3gp','3gp2','3g2','3gpp','3gpp2','ogg','oga','ogv','ogx','wmv','wma','asf','webm','flv','avi','qt'}
Allowed_Encoding = {'pickle'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "abc"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file1(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Allowed_Encoding


@app.route('/', methods = ['GET','POST'])
def Connect():
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
      print(filename,"File Saved")
      return render_template("home.html")
    else:
        return render_template("Train.html")

@app.route('/Video', methods = ['GET','POST'])
def video():
    if request.method == 'POST':
        print("")
        print("")
        print(request.files)
        print("")
        print("")

        try:
            vfile = request.files['file']
            vfilename = secure_filename(vfile.filename)
            if not (allowed_file(vfile.filename)):
                flash('Extension Not Allowed')
                return render_template("home.html")
            vfile.save(os.path.join(app.config['UPLOAD_FOLDER'], vfilename))      
            print(vfilename,"  File Saved")
        except:
            flash('Invalid')
        
        try:
            enfile = request.files['vfile']
            filename = secure_filename(enfile.filename)
            if not (allowed_file1(vfile.filename)):
                flash('Extension Not Allowed')
                return render_template("home.html")
            enfile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))      
            print(filename,"  File Saved")
        except:
            print("No Encoding Recived")

        return render_template("home.html")
    else:
        return render_template("Train.html")

if __name__=="__main__":
	app.run(debug = True)