from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
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
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the PDF
        result = process_pdf(filepath)
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'result': result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:report_identifier>', methods=['GET'])
def get_report(report_identifier):
    try:
        # Check if the identifier is a PDF file
        if report_identifier.lower().endswith('.pdf'):
            # Search for the corresponding JSON file in all report type folders
            for report_type in os.listdir('extracted_reports'):
                report_type_path = os.path.join('extracted_reports', report_type)
                if os.path.isdir(report_type_path):
                    for filename in os.listdir(report_type_path):
                        if filename.endswith('.json'):
                            with open(os.path.join(report_type_path, filename), 'r') as f:
                                try:
                                    # Parse the JSON data
                                    report_data = json.loads(f.read())
                                    
                                    # Create a structured response
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
                                    
                                    # Organize parameters by category
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
                                    
                                except json.JSONDecodeError:
                                    continue
            
            return jsonify({'error': 'Report not found'}), 404
            
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)