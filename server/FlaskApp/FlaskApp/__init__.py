import os
from flask import Flask, jsonify, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import process

UPLOAD_FOLDER = 'files'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_AS_ASCII'] = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('process_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

from shutil import copyfile

@app.route('/process')
def process_file():
    filename = request.args.get('filename', '')
    try:
        response = process.process('files/' + filename)
    except Exception:
        return jsonify({ 'error': 'File not found' })
    copyfile("files/" + filename, "static/" + filename)
    return  ''' 
    <!doctype html>
    <img src="static/''' + filename + '''" width="960" height="540"/>
    <br>
    <br>
    <h3>''' + response + '''</h3>
    '''

if __name__ == "__main__":
    app.run()