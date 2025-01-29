from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify
import os
from werkzeug.utils import secure_filename
import subprocess
import shutil
import sys
import json
import csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GIF_FOLDER'] = 'static/gifs'
app.secret_key = 'your_secret_key'

# Ensure upload and gif folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GIF_FOLDER'], exist_ok=True)

# Function to convert .tsp file to .csv
def tsp_to_csv(tsp_file):
    with open(tsp_file, 'r') as file:
        lines = file.readlines()
        with open('small_tsp.csv', 'w') as file:
            writer = csv.writer(file)
            for line in lines:
                if line[0].isdigit():
                    _, x, y = line.split()
                    writer.writerow([float(x), float(y)])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file_immediately', methods=['POST'])
def upload_file_immediately():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Store the original filename in session
    session['original_filename'] = filename

    # Check file type and convert if necessary
    if filename.endswith('.tsp'):
        tsp_to_csv(file_path)  # Convert .tsp to .csv
        file_path = 'small_tsp.csv'  # Use the converted CSV file
        session['converted_file_path'] = file_path
    else:
        session['converted_file_path'] = file_path

    return jsonify({'success': True, 'filename': filename})

@app.route('/upload', methods=['POST'])
def upload_file():
    # Get the filename from the session
    filename = session.get('original_filename')
    file_path = session.get('converted_file_path')
    
    if not filename or not file_path:
        return redirect(url_for('index'))

    algorithm = request.form.get('algorithm')
    if algorithm == 'ACO':
        gif_name = 'abc.gif'
        script = 'aco.py'
    elif algorithm == 'GA':
        gif_name = 'xyz.gif'
        script = 'ga.py'
    else:
        return redirect(request.url)

    # Run the selected algorithm script with the file path
    subprocess.Popen(['python3', script, file_path])

    return redirect(url_for('loading', gif_name=gif_name))

@app.route('/show_stats', methods=['POST'])
def show_stats():
    # Get the file path from session
    file_path = session.get('converted_file_path')
    
    if not file_path:
        return redirect(url_for('index'))
    
    # Run statistical analysis
    subprocess.run(['python3', 'statistical_inferences.py', file_path])
    
    # Prepare stats data for the template
    stats_data = []
    
    # Add text-based statistics
    text_files = [
        'data_info.txt', 'central_tendency.txt', 'dispersion_measures.txt',
        'correlation_analysis.txt', 'outliers.txt', 'morans_i.txt',
        'distance_metrics.txt', 'convex_hull.txt', 'nearest_neighbor_stats.txt'
    ]
    
    image_files = [
        'histograms.png', 'scatter_plot.png', 'outlier_detection.png',
        'density_plot.png', 'kmeans_clustering.png', 'dbscan_clustering.png',
        'convex_hull.png', 'nearest_neighbor_distances.png'
    ]
    
    # Read text files
    for filename in text_files:
        filepath = os.path.join('static/inferences', filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
                stats_data.append({
                    'type': 'text',
                    'content': content
                })
    
    # Add image files
    for filename in image_files:
        filepath = os.path.join('static/inferences', filename)
        if os.path.exists(filepath):
            stats_data.append({
                'type': 'image',
                'content': url_for('static', filename=f'inferences/{filename}')
            })
    
    return render_template('stats.html', stats_data=stats_data)

@app.route('/loading/<gif_name>')
def loading(gif_name):
    return render_template('loading.html', gif_name=gif_name)

@app.route('/result/<gif_name>')
def result(gif_name):
    return render_template('result.html', gif_name=gif_name)

if __name__ == '__main__':
    app.run(debug=True)