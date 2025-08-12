from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_maintenance():
    return send_from_directory(app.static_folder, 'index2.html')

if __name__ == '__main__':
    app.run(port=8080)