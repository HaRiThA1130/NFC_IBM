from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_folder='.')

@app.route('/')
@app.route('/index')
@app.route('/index.html')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/aboutus')
@app.route('/aboutus.html')
def aboutus():
    return send_from_directory('.', 'aboutus.html')

@app.route('/facultylogin')
@app.route('/faculty-login')
@app.route('/facultylogin.html')
def faculty_login():
    return send_from_directory('.', 'faculty-login.html')

@app.route('/faculty-dashboard-fixed.html')
def faculty_dashboard():
    return send_from_directory('.', 'faculty-dashboard-fixed.html')

@app.route('/<path:filename>')
def serve_file(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
