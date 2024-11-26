from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename # Add this import
from test_identification import process_pdf

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add a helper function to clean filenames
def clean_filename(filename):
    # Remove special characters and replace spaces with underscores
    cleaned = secure_filename(filename)
    # Convert to lowercase for consistency
    return cleaned.lower()

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Clean and secure the filename
        filename = clean_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the PDF
        result = process_pdf(filepath)
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'filename': filename,
            'result': result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:report_identifier>', methods=['GET'])
def get_report(report_identifier):
    try:
        # Check if the identifier is a PDF file
        if report_identifier.lower().endswith('.pdf'):
            # Clean and standardize the filename
            base_filename = os.path.splitext(report_identifier)[0]
            base_filename = clean_filename(base_filename)
            
            print(f"Looking for report with base filename: {base_filename}")
            
            # Search for the corresponding JSON file in all report type folders
            for report_type in os.listdir('extracted_reports'):
                report_type_path = os.path.join('extracted_reports', report_type)
                print(f"Searching in directory: {report_type_path}")
                
                if os.path.isdir(report_type_path):
                    for filename in os.listdir(report_type_path):
                        print(f"Checking file: {filename}")
                        
                        # Convert both filenames to lowercase for comparison
                        if filename.endswith('.json') and base_filename in filename.lower():
                            json_path = os.path.join(report_type_path, filename)
                            print(f"Found matching file: {json_path}")
                            
                            with open(json_path, 'r') as f:
                                try:
                                    report_data = json.loads(f.read())
                                    
                                    structured_report = {
                                        'original_filename': report_identifier,
                                        'processed_filename': filename,
                                        'report_type': report_type,
                                        'test_type': report_data.get('test', ''),
                                        'hospital_info': {
                                            'name': report_data.get('hospital_lab_name', ''),
                                            'location': report_data.get('hospital_lab_location', '')
                                        },
                                        'patient_name': report_data.get('Patient Name', ''),
                                        'parameters': {}
                                    }
                                    
                                    if 'parameters' in report_data:
                                        for param in report_data['parameters']:
                                            category = param['parameter']
                                            if category not in structured_report['parameters']:
                                                structured_report['parameters'][category] = []
                                            
                                            param_data = {
                                                'sub_parameter': param.get('sub_parameter', ''),
                                                'value': param.get('value', ''),
                                                'units': param.get('units', '')
                                            }
                                            structured_report['parameters'][category].append(param_data)
                                    
                                    return jsonify(structured_report), 200
                                    
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing JSON: {e}")
                                    continue
            
            return jsonify({
                'error': f'No processed report found for {report_identifier}',
                'searched_filename': base_filename
            }), 404
            
        else:
            # Handle report type request (existing functionality)
            reports_path = os.path.join('extracted_reports', report_identifier)
            
            if not os.path.exists(reports_path):
                return jsonify({'error': 'Report type not found'}), 404
            
            reports = []
            for filename in os.listdir(reports_path):
                if filename.endswith('.json'):
                    with open(os.path.join(reports_path, filename), 'r') as f:
                        try:
                            report_data = json.loads(f.read())
                            structured_report = {
                                'filename': filename,
                                'test_type': report_data.get('test', ''),
                                'hospital_info': {
                                    'name': report_data.get('hospital_lab_name', ''),
                                    'location': report_data.get('hospital_lab_location', '')
                                },
                                'patient_name': report_data.get('Patient Name', ''),
                                'parameters': {}
                            }
                            
                            if 'parameters' in report_data:
                                for param in report_data['parameters']:
                                    category = param['parameter']
                                    if category not in structured_report['parameters']:
                                        structured_report['parameters'][category] = []
                                    
                                    param_data = {
                                        'sub_parameter': param.get('sub_parameter', ''),
                                        'value': param.get('value', ''),
                                        'units': param.get('units', '')
                                    }
                                    structured_report['parameters'][category].append(param_data)
                            
                            reports.append(structured_report)
                            
                        except json.JSONDecodeError:
                            continue
            
            return jsonify({
                'total_reports': len(reports),
                'report_type': report_identifier,
                'reports': reports
            }), 200
            
    except Exception as e:
        print(f"Error in get_report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/available-reports', methods=['GET'])
def list_available_reports():
    try:
        available_reports = []
        for report_type in os.listdir('extracted_reports'):
            report_type_path = os.path.join('extracted_reports', report_type)
            if os.path.isdir(report_type_path):
                for filename in os.listdir(report_type_path):
                    if filename.endswith('.json'):
                        available_reports.append({
                            'report_type': report_type,
                            'filename': filename
                        })
        
        return jsonify({
            'total_reports': len(available_reports),
            'reports': available_reports
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)