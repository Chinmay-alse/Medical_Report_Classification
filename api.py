from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
from test_identification import process_pdf
import psycopg2
from psycopg2.extras import Json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Database configuration
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_filename(filename):
    cleaned = secure_filename(filename)
    return cleaned.lower()

def save_to_database(filename, processed_data, report_type, patient_name=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO report_results 
        (filename, processed_data, report_type, patient_name, created_at, status)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        if isinstance(processed_data, str):
            try:
                processed_data = json.loads(processed_data)
            except json.JSONDecodeError:
                processed_data = {"raw_text": processed_data}
        
        cur.execute(insert_query, (
            filename,
            Json(processed_data),
            report_type,
            patient_name,
            datetime.now(),
            'completed'
        ))
        
        result_id = cur.fetchone()[0]
        conn.commit()
        
        cur.close()
        conn.close()
        
        return result_id
    
    except Exception as e:
        print(f"Error saving to database: {e}")
        if 'conn' in locals():
            conn.close()
        return None

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
        
        if result:
            try:
                # Parse the result
                result_data = json.loads(result) if isinstance(result, str) else result
                report_type = result_data.get('test', 'Unknown')
                patient_name = result_data.get('Patient Name', None)
                
                # Save to database
                db_id = save_to_database(
                    filename=filename,
                    processed_data=result_data,
                    report_type=report_type,
                    patient_name=patient_name
                )
                
                if db_id:
                    # Return only report_id, message, and filename
                    return jsonify({
                        'report_id': db_id,
                        'message': 'File uploaded and processed successfully',
                        'filename': filename
                    }), 200
                else:
                    return jsonify({
                        'error': 'Failed to save to database',
                        'filename': filename
                    }), 500
                
            except Exception as e:
                return jsonify({
                    'error': f'Error processing file: {str(e)}',
                    'filename': filename
                }), 500
        
        return jsonify({
            'error': 'Failed to process file',
            'filename': filename
        }), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report/<int:report_id>', methods=['GET'])
def get_report_by_id(report_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT filename, processed_data, created_at, report_type, patient_name 
            FROM report_results 
            WHERE id = %s
        """, (report_id,))
        
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            processed_data = result[1]  # This is the JSONB data from the database
            
            # Structure the response
            report_data = {
                'report_type': result[3],
                'patient_name': result[4],
                'hospital_info': {
                    'name': processed_data.get('hospital_lab_name', ''),
                    'location': processed_data.get('hospital_lab_location', '')
                },
                'parameters': processed_data.get('parameters', {})
            }
            
            return jsonify(report_data), 200
        
        return jsonify({'error': 'Report not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports', methods=['GET'])
def list_reports():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Add pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        offset = (page - 1) * per_page
        
        # Get total count
        cur.execute("SELECT COUNT(*) FROM report_results")
        total_reports = cur.fetchone()[0]
        
        # Get paginated results with basic information
        cur.execute("""
            SELECT id, filename, report_type, patient_name, created_at 
            FROM report_results 
            ORDER BY created_at DESC 
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        
        reports = []
        for row in cur.fetchall():
            reports.append({
                'report_id': row[0],
                'filename': row[1],
                'report_type': row[2],
                'patient_name': row[3],
                'created_at': row[4].isoformat()
            })
        
        cur.close()
        conn.close()
        
        return jsonify({
            'total_reports': total_reports,
            'page': page,
            'per_page': per_page,
            'reports': reports
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)