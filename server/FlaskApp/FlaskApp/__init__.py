from flask import Flask, request, jsonify
import process

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

@app.route('/')
def hello():
    return 'Hello, I love Digital Ocean!'

@app.route('/process')
def process_file():
    filename = request.args.get('filename', '')
    try:
        response = process.process('files/' + filename)
    except Exception:
        return jsonify({ 'error': 'File not found' })
    return jsonify({ 'response': response })

if __name__ == "__main__":
    app.run()