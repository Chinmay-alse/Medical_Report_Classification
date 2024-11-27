import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import base64
from pdf2image import convert_from_path
import os
import json
import pypdfium2 as pdfium
import re
from openai import AzureOpenAI
import time
import cv2
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import psycopg2
from datetime import datetime

def create_db_connection():
    try:
        connection = psycopg2.connect(
            database="your_database_name",
            user="your_username",
            password="your_password",
            host="your_host",
            port="5432"
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Set up Tesseract path and remove image size limit
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Image.MAX_IMAGE_PIXELS = 500000000

# Azure OpenAI client initialization
api_base = "https://azure-isv-success-in.openai.azure.com/"
api_key = "7c90d344cb524b9885202a7603641589"
deployment_name = "gpt-4o"
api_version = "2024-06-01"

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}openai/deployments/{deployment_name}",
)
# Define classification keywords for CBC, Serum, Endoscopy, and Clinical Biochemistry
cbc_keywords = [
    "haemoglobin", "red blood cell count", "packed cell volume", "haematocrit",
    "Mean corpuscular volume", "mean corpuscular haemoglobin", "mean corpuscular haemoglobin concentration",
    "Platelet count", "leukocyte count", "white blood cell count", "RBC", "WBC", "MCV", "MCH", "MCHC",
    "WBC COUNT", "RBC COUNT", "BASOPHIL ABS", "COMPLETE BLOOD COUNT", "COMPLETE BLOOD COUNT REPORT","Complete Blood Count","Mean Corpuscular Hemoglobin Concentration (MCHC)","Leukocytes"
]

serum_keywords = [
    "bio markers", "Carcinoembryonic Antigen", "CEA", "Alpha-Fetoprotein", "Prostate Specific Antigen", 
    "Alanine Transaminase", "Aspartate Transaminase", "Blood Urea Nitrogen", "serum biomarkers"
]


endoscopy_keywords = [
    "endoscopy", "colonoscopy", "gastroscopy", "sigmoidoscopy", "biopsy", "gastroscopy", "oesophagus",
    "stomach", "colon", "rectum", "Erosion", "Ulcer", "Polyps", "Lesions", "Inflammation", "Diverticulosis",
    "Strictures", "Stenosis", "Haemorrhage", "Tumour", "Varices", "Biopsy","Colonoscopy","endoscopy", "rectum", "sigmoid", "hemorrhoiods", "biopsy forceps", "dimunitive polyps", "Ulcer", "Stricture","tumor","cecum", "Normal vascular pattern", "Ilocecal opening", "Terminal Ileum"
]

clinical_biochemistry_keywords = [
    "Biochemistry","LFT", "Electrolytes", "KFT", "Blood sugar", "Thyroid profile", "Creatinine", "Urea","uric acid", "bilirubin",
    "albumin", "protein", "sodium", "potassium", "calcium", "phosphate", "blood urea nitrogen", "ALT", "AST", "serum creatinine","crp","ldh",
    "BUN", "Cholesterol", "Triglycerides", "Bilirubin", "Hepatitis C", "Thyroid", "Insulin", "Diabetes", "creatinine clearance","eGFR","BIOCHEMISTRY",
    "Clinical Biochemistry", "Liver function test", "Kidney function test", "HbA1c","T3","T4","Triiodothyronine","Thyroxine","TSH","Renal Function Test","Liver function test","Lipid Profile","Glucose tolerance","fasting blood sugar","Sodium", "Potassium", "Chloride", "Fasting blood sugar", "HbA1c", "post pranadial blood sugar", "Random blood sugar", "Total Cholesterol", "HDL", "LDL", "Triglycerides", "VLDL", "ALT", "AST", "Bilirubin", "Albumin", "Alkaline phosphatase", "Direct bilirubin", "Globulin", "Indirect bilirubin", "Serum total protein", "Gamma-glutamyl Transferase", "Total bilirubin", "Creatinine", "Urea", "BUN", "Blood urea nitrogen", "Creatinine clearance", "eGFR", "Serum creatinine", "Uric acid", "T3", "T4", "TSH", "CRP", "LDH", "Uric Acid", "CMV", "Hepatitis B antigen", "Hepatitis C", "HIV"
]

import psycopg2
from psycopg2.extras import Json

def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
                
def save_to_database(filename, processed_data, report_type, patient_name=None):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO report_results 
        (filename, processed_data, report_type, patient_name, status)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        # Convert processed_data to JSON if it's a string
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
            'completed'
        ))
        
        result_id = cur.fetchone()[0]
        conn.commit()
        
        print(f"Successfully saved to database with ID: {result_id}")
        
        cur.close()
        conn.close()
        
        return result_id
        
    except Exception as e:
        print(f"Error saving to database: {e}")
        if 'conn' in locals():
            conn.close()
        return None
    
def load_system_prompt(report_type):
    # Path to the system prompts directory
    prompt_file_path = os.path.join("System_prompts", f"{report_type}_prompt.txt")
    
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r") as file:
            return file.read()
    else:
        return "Assistant is a large language model trained by OpenAI. Enhance the extracted medical text and provide it in JSON format."

def generate_prompt(extracted_text, report_type):
    system_prompt = load_system_prompt(report_type)  # Load prompt dynamically based on report type
    user_prompt = extracted_text
    return system_prompt, user_prompt

# Function to send the prompt to the Azure LLM
def extract_parameters_from_text(text, report_type):
    """
    Extract parameters from the OCR text based on report type.
    Returns a set of found parameters.
    """
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    found_parameters = set()
    
    parameter_patterns = {
        "CBC": {
    "haemoglobin": r"h[ae]moglobin.*?(\d+\.?\d*)",
    "wbc": r"wbc.*?(\d+\.?\d*)",
    "platelets": r"platelet.*?(\d+\.?\d*)",
    "rbc": r"rbc.*?(\d+\.?\d*)",
    "mcv": r"mcv.*?(\d+\.?\d*)",
    "mch": r"mch.*?(\d+\.?\d*)",
    "mchc": r"mchc.*?(\d+\.?\d*)",
    "hematocrit": r"(?:hematocrit|pcv).*?(\d+\.?\d*)",
    "rdw": r"rdw.*?(\d+\.?\d*)",
    "neutrophils": r"neutrophils.*?(\d+\.?\d*)",
    "lymphocyte": r"lymphocyte.*?(\d+\.?\d*)",
    "eosinophil": r"eosinophil.*?(\d+\.?\d*)",
    "basophil": r"basophil.*?(\d+\.?\d*)",
    "wbc_diff": r"(?:wbc|leukocyte).*?(?:differential|diff).*?(\d+\.?\d*)"
}
,
        "Clinical_Biochemistry": {
            "glucose": r"glucose.*?(\d+\.?\d*)",
            "creatinine": r"creatinine.*?(\d+\.?\d*)",
            "urea": r"urea.*?(\d+\.?\d*)",
            "sodium": r"sodium.*?(\d+\.?\d*)",
            "potassium": r"potassium.*?(\d+\.?\d*)"
        },
    "Serum": {
    "ca_125": r"ca[- ]?125.*?(\d+\.?\d*)",
    "ca_19_9": r"ca[- ]?19[- ]?9.*?(\d+\.?\d*)",
    "ca_15_3": r"ca[- ]?15[- ]?3.*?(\d+\.?\d*)",
    "cea": r"(?:cea|carcinoembryonic antigen).*?(\d+\.?\d*)"
}
,
    "IHC": {
        "er": r"(?:er|estrogen receptor).*?(\d+%|\d+\.?\d*)",
        "pr": r"(?:pr|progesterone receptor).*?(\d+%|\d+\.?\d*)",
        "her2": r"(?:her2|her-2).*?(positive|negative|\+|\-|[0-3]\+?)",
        "ki67": r"ki-?67.*?(\d+%|\d+\.?\d*)",
        "p53": r"p53.*?(positive|negative|\+|\-|\d+%)",
        "cd3": r"cd3.*?(positive|negative|\+|\-|\d+%)",
        "cd20": r"cd20.*?(positive|negative|\+|\-|\d+%)",
        "cd30": r"cd30.*?(positive|negative|\+|\-|\d+%)",
        "cd45": r"cd45.*?(positive|negative|\+|\-|\d+%)",
        "ck": r"(?:ck|cytokeratin).*?(positive|negative|\+|\-|\d+%)",
        "ck7": r"ck7.*?(positive|negative|\+|\-|\d+%)",
        "ck20": r"ck20.*?(positive|negative|\+|\-|\d+%)"
    },
    "Endoscopy": {
        "procedure_type": r"(?:procedure|examination).*?(colonoscopy|gastroscopy|sigmoidoscopy)",
        "scope_insertion": r"scope.*?inserted.*?(.*?)(?:\.|\n)",
        "mucosa": r"mucosa.*?(normal|abnormal|erythematous|inflamed|pale)",
        "ulcers": r"ulcer.*?(present|absent|seen|noted|multiple|single)",
        "polyps": r"polyp.*?(present|absent|seen|noted|multiple|single)",
        "size": r"(?:size|measuring).*?(\d+\.?\d*\s*(?:mm|cm))",
        "location": r"(?:located|found|seen).*?(antrum|body|fundus|duodenum|rectum|sigmoid|descending|transverse|ascending|cecum)",
        "biopsy": r"biops.*?(taken|performed|done|obtained)",
        "bleeding": r"bleeding.*?(present|absent|active|noted)",
        "impression": r"impression.*?(.*?)(?:\.|\n)"
    }
    }
    
    
    if report_type in parameter_patterns:
        for param, pattern in parameter_patterns[report_type].items():
            if re.search(pattern, text_lower):
                found_parameters.add(param)
    
    return found_parameters

def get_expected_parameters(report_type):
    """
    Get the expected parameters for a given report type.
    Returns a set of expected parameters.
    """
    expected_parameters = {
    "CBC": {"haemoglobin", "rbc", "wbc", "platelets", "mcv", "mch", "mchc"},
    "Clinical_Biochemistry": {"glucose", "creatinine", "urea", "sodium", "potassium"},
    "Serum": {"cea", "afp", "psa", "ca_125", "ca_19_9", "ca_15_3", "tsh", "t3", "t4", 
              "vitamin_b12", "vitamin_d", "ferritin"},
    "IHC": {"er", "pr", "her2", "ki67", "p53", "cd3", "cd20", "cd30", "cd45", "ck", 
            "ck7", "ck20"},
    "Endoscopy": {"procedure_type", "scope_insertion", "mucosa", "ulcers", "polyps", 
                  "size", "location", "biopsy", "bleeding", "impression"}
}
    
    return expected_parameters.get(report_type, set())

def check_missing_parameters(extracted_text, report_type):
    """
    Check which parameters are missing from the extracted text.
    Returns a set of missing parameters.
    """
    expected_params = get_expected_parameters(report_type)
    found_params = extract_parameters_from_text(extracted_text, report_type)
    
    return expected_params - found_params

def enhance_text_with_groq(extracted_text, report_type, image_paths=None):
    """
    Enhanced version that only uses images when parameters are missing.
    """
    missing_params = check_missing_parameters(extracted_text, report_type)
    
    if not missing_params:
        print("No missing parameters detected. Processing without images.")
        # Process without images
        system_prompt, user_prompt = generate_prompt(extracted_text, report_type)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        print(f"Missing parameters detected: {missing_params}")
        # Process with images
        system_prompt, _ = generate_prompt(extracted_text, report_type)
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        content = [
            {
                "type": "text",
                "text": f"Add the following missing parameters to the extracted text: {', '.join(missing_params)}. Use only the numerical values from the images. Return the response in valid JSON format."
            }
        ]
        
        if image_paths:
            for image_path in image_paths:
                base64_image = encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                })
        
        messages.append({
            "role": "user",
            "content": content
        })

    try:
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1
        )
        
        enhanced_text = response.choices[0].message.content
        print("\nReceived response from LLM")
        return enhanced_text
            
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

def save_text_to_json(extracted_text, report_type, original_filename):
    if not extracted_text:
        print("No text to save")
        return
    
    # Create a directory for the report type if it doesn't exist
    folder_path = os.path.join("extracted_reports", report_type)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Clean and standardize the filename
    base_filename = os.path.splitext(original_filename)[0]
    base_filename = secure_filename(base_filename)
    
    # Create a unique filename using the original filename and timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder_path, f"{base_filename}_{timestamp}.json")
    
    try:
        # Clean the text
        cleaned_text = extracted_text.strip()
        
        # Remove markdown code blocks if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        # Write the cleaned text directly to the file
        with open(filename, 'w', encoding='utf-8', newline='\n') as file:
            file.write(cleaned_text)
        
        print(f"\nSuccessfully saved text to {filename}")
        
    except Exception as e:
        print(f"Error saving to file: {e}")
        
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

from PIL import Image, ImageFilter, ImageEnhance

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    # Apply a median filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    # Sharpen the image to make edges more defined
    image = image.filter(ImageFilter.SHARPEN)
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Convert to binary image with adaptive thresholding
    image = image.point(lambda x: 0 if x < 128 else 255, '1')
    return image



def resize_if_necessary(image_path):
    img = Image.open(image_path)
    width, height = img.size
    current_pixels = width * height
    max_pixels = 33177600
    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img_resized = img.resize((new_width, new_height))
        resized_path = "resized_image.png"
        img_resized.save(resized_path)
        return resized_path
    return image_path

from PIL import Image

# Function to convert PDF to images using two options with fallback logic
def convert_pdf_to_images_option1(pdf_path):
    try:
        # Convert all pages
        images = convert_from_path(pdf_path)
        image_paths = []
        
        # Create a standardized name base for the images
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Save each page as a separate image
        for i, image in enumerate(images):
            image_path = f"{base_name}_page_{i+1}_converted.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        
        return image_paths
    except Exception as e:
        print(f"Option 1 failed with error: {e}")
        return None

def convert_pdf_to_images_option2(pdf_path, scale=300/72):
    try:
        pdf_file = pdfium.PdfDocument(pdf_path)
        # Get all pages
        page_indices = [i for i in range(len(pdf_file))]
        
        # Create a standardized name base for the images
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Render pages
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=scale,
        )
        
        image_paths = []
        # Convert generator to list and save all images
        images = list(renderer)
        for i, image in enumerate(images):
            image_path = f"{base_name}_page_{i+1}_converted.png"
            image.save(image_path, format='PNG')
            image_paths.append(image_path)
            
        return image_paths
        
    except Exception as e:
        print(f"Option 2 failed with error: {e}")
        return None

def convert_pdf_to_images(pdf_path):
    # Create output directory if it doesn't exist
    output_dir = "converted_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get base name for the images
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    final_image_paths = []
    
    # Attempt Option 1
    try:
        image_paths = convert_pdf_to_images_option1(pdf_path)
        if image_paths and len(image_paths) > 0:
            # Process each image
            insufficient_text = False
            temp_image_paths = []
            
            for i, img_path in enumerate(image_paths):
                text = extract_text_from_image(img_path)
                word_count = len(text.split())
                print(f"Option 1 - Page {i+1} word count: {word_count}")
                
                if word_count < 14:  # Check if word count is less than 14
                    insufficient_text = True
                    print(f"Page {i+1} has insufficient text ({word_count} words)")
                    break
                
                final_path = os.path.join(output_dir, f"{base_name}_page_{i+1}_converted.png")
                if os.path.exists(img_path):
                    os.replace(img_path, final_path)
                    temp_image_paths.append(final_path)
            
            # Clean up original images
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
            
            if not insufficient_text:
                return temp_image_paths
            else:
                print("Switching to Option 2 due to insufficient text")
                # Clean up Option 1 converted images
                for path in temp_image_paths:
                    if os.path.exists(path):
                        os.remove(path)
    
    except Exception as e:
        print(f"Error in Option 1: {e}")
    
    # Use Option 2
    try:
        print("Attempting Option 2 conversion")
        image_paths = convert_pdf_to_images_option2(pdf_path)
        if image_paths and len(image_paths) > 0:
            # Move each image to the final location
            for i, img_path in enumerate(image_paths):
                final_path = os.path.join(output_dir, f"{base_name}_page_{i+1}_converted.png")
                if os.path.exists(img_path):
                    os.replace(img_path, final_path)
                    final_image_paths.append(final_path)
            return final_image_paths
    except Exception as e:
        print(f"Error in Option 2: {e}")
    
    return None

def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 --oem 3')
    print(text)
    return text
    
def classify_report_with_groq(corrected_text, system_prompt_content):
    """
    Classifies the report using the Groq LLM.
    """
    try:
        classification_response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_content,
                },
                {
                    "role": "user",
                    "content": corrected_text,
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        
        classification_result = classification_response.choices[0].message.content
        print(f"Raw Classification Result: {classification_result}")
        
        # Clean up the response and ensure it's valid JSON
        cleaned_result = classification_result.strip()
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:-3]
        
        # Parse the JSON
        result_json = json.loads(cleaned_result)
        
        # Map category or type to standardized report_type
        report_type = None
        
        # Check various possible keys
        if "category" in result_json:
            report_type = result_json["category"]
        elif "type" in result_json:
            report_type = result_json["type"]
        elif "report_type" in result_json:
            report_type = result_json["report_type"]
            
        # Standardize the report type
        if report_type:
            # Map common variations to standard names
            report_type_mapping = {
                "SERUM": "Serum_Analysis",
                "CBC": "CBC",
                "ENDOSCOPY": "Endoscopy",
                "CLINICAL BIOCHEMISTRY": "Clinical_Biochemistry",
                "BIOCHEMISTRY": "Clinical_Biochemistry"
            }
            
            report_type = report_type.upper()  # Convert to uppercase for consistent matching
            standardized_type = report_type_mapping.get(report_type, report_type)
            
            # Create standardized response
            standardized_response = {
                "report_type": standardized_type,
                "confidence_score": result_json.get("confidence", result_json.get("match_percentage", 0)) if isinstance(result_json.get("confidence"), (int, float)) else 0.5,
                "keywords_identified": result_json.get("matched_keywords", [])
            }
            
            return standardized_response
            
        return {"report_type": "Unknown", "confidence_score": 0, "keywords_identified": []}
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return {"report_type": "Unknown", "confidence_score": 0, "keywords_identified": []}

def read_system_prompt(prompt_file):
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        # Provide a default system prompt if file is not found
        return

def document_report_processing(pdf_path, classification_result, enhanced_text, processing_time=None):
    """
    Documents the report processing details in a separate log file.
    
    Args:
        pdf_path (str): Path to the original PDF file
        classification_result (dict): Result from the classification
        enhanced_text (dict): Enhanced text from Groq
        processing_time (float, optional): Total processing time in seconds
    """
    # Create logs directory if it doesn't exist
    log_dir = "processing_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a timestamp for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"report_processing_log_{timestamp}.txt")
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Write header
        log_file.write("=" * 50 + "\n")
        log_file.write("MEDICAL REPORT PROCESSING LOG\n")
        log_file.write("=" * 50 + "\n\n")
        
        # Document input file details
        log_file.write("INPUT FILE DETAILS:\n")
        log_file.write(f"File Name: {os.path.basename(pdf_path)}\n")
        log_file.write(f"Processing Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if processing_time:
            log_file.write(f"Total Processing Time: {processing_time:.2f} seconds\n")
        log_file.write("\n")
        
        # Document classification details
        log_file.write("CLASSIFICATION DETAILS:\n")
        log_file.write(f"Detected Report Type: {classification_result.get('report_type', 'Unknown')}\n")
        log_file.write(f"Confidence Score: {classification_result.get('confidence_score', 'N/A')}\n")
        if 'keywords_identified' in classification_result:
            log_file.write("Keywords Identified:\n")
            for keyword in classification_result['keywords_identified']:
                log_file.write(f"- {keyword}\n")
        log_file.write("\n")
        
        # Document enhancement details
        log_file.write("ENHANCEMENT DETAILS:\n")
        if isinstance(enhanced_text, dict):
            for key, value in enhanced_text.items():
                if isinstance(value, (list, dict)):
                    log_file.write(f"{key}:\n{json.dumps(value, indent=2)}\n")
                else:
                    log_file.write(f"{key}: {value}\n")
        else:
            log_file.write("Enhanced text format: Unknown\n")
        
        # Write footer
        log_file.write("\n" + "=" * 50 + "\n")
        log_file.write("END OF LOG\n")
        log_file.write("=" * 50 + "\n")
    
    print(f"Processing log saved to: {log_filename}")
    
def enhance_text_with_azure(extracted_text, report_type, image_paths=None):
    """
    Enhanced version that uses Azure OpenAI for text enhancement with image support.
    """
    missing_params = check_missing_parameters(extracted_text, report_type)
    
    if not missing_params:
        print("No missing parameters detected. Processing without images.")
        # Process without images
        system_prompt, user_prompt = generate_prompt(extracted_text, report_type)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model=deployment_name,  # Use your Azure OpenAI deployment name
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                top_p=0.95,
                response_format={"type": "json_object"}
            )
            
            enhanced_text = response.choices[0].message.content
            print("\nReceived response from Azure OpenAI")
            return enhanced_text
            
        except Exception as e:
            print(f"Error in API call: {e}")
            return None
            
    else:
        print(f"Missing parameters detected: {missing_params}")
        system_prompt, _ = generate_prompt(extracted_text, report_type)
        
        image_analysis_prompt = f"""
        Analyze the medical report and images to extract the following missing parameters: {', '.join(missing_params)}.
        
        Current extracted text:
        {extracted_text}
        
        Instructions:
        1. Focus on finding exact numerical values for the missing parameters including patient name and hospital name and location.
        2. DO NOT add any parameters not listed in the system_prompt
        3. Look for values in both the text and images
        4. Ensure all values include their units
        5. Return the results in valid JSON format
        6. Include reference ranges where available
        7. Verify values against both text and image data
        8. Maintain same structure as mentioned in the system_prompt
        
        Return the complete report data including both existing and newly found parameters.
        """
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        content = [
            {
                "type": "text",
                "text": image_analysis_prompt
            }
        ]

        if image_paths:
            compressed_image_paths = []
            for image_path in image_paths:
                compressed_path = compress_image_for_api(image_path)
                if compressed_path:
                    compressed_image_paths.append(compressed_path)
            
            for compressed_path in compressed_image_paths:
                try:
                    base64_image = encode_image(compressed_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                except Exception as e:
                    print(f"Error processing image {compressed_path}: {e}")
                finally:
                    # Clean up compressed image
                    if os.path.exists(compressed_path):
                        os.remove(compressed_path)

        messages.append({
            "role": "user",
            "content": content
        })

        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                top_p=0.95,
                response_format={"type": "json_object"}
            )
            
            enhanced_text = response.choices[0].message.content
            print("\nReceived response from Azure OpenAI")
            
            try:
                json_response = json.loads(enhanced_text)
                return json.dumps(json_response, indent=2)
            except json.JSONDecodeError:
                print("Warning: Response was not in valid JSON format. Returning raw response.")
                return enhanced_text
                
        except Exception as e:
            print(f"Error in Azure OpenAI API call: {e}")
            return None
        
def compress_image_for_api(image_path, max_size_mb=19):
    """
    Compress and resize image to ensure it's under the API size limit
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Initial quality
            quality = 95
            output_path = f"compressed_{os.path.basename(image_path)}"
            
            while True:
                # Save with current quality
                img.save(output_path, 'JPEG', quality=quality)
                
                # Check file size
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                if size_mb <= max_size_mb:
                    return output_path
                
                # Reduce quality if file is still too large
                quality -= 10
                
                # If quality is too low, try resizing
                if quality < 30:
                    # Resize image
                    width, height = img.size
                    new_width = int(width * 0.8)
                    new_height = int(height * 0.8)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    quality = 95  # Reset quality
                
                # Prevent infinite loop
                if quality < 20 and size_mb > max_size_mb:
                    raise ValueError("Unable to compress image sufficiently")
                
    except Exception as e:
        print(f"Error compressing image: {e}")
        return None


def process_pdf(pdf_path):
    start_time = time.time()
    
    original_filename = os.path.basename(pdf_path)
    
    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path)
    
    if not image_paths or len(image_paths) == 0:
        print("Failed to convert PDF to images")
        return
    
    if not image_paths or len(image_paths) == 0:
        print("Failed to convert PDF to images")
        return
    
    print(f"Successfully converted PDF to {len(image_paths)} images")
    
    # Extract text from all images
    combined_text = ""
    for i, image_path in enumerate(image_paths):
        page_text = extract_text_from_image(image_path)
        combined_text += f"\nPage {i+1}:\n{page_text}"
    
    print(f"Total extracted text length: {len(combined_text.split())}")
    
    # Read classification system prompt
    classification_system_prompt = read_system_prompt("classify_sys_prompt.txt")
    
    # Classify the combined text
    classification_result = classify_report_with_groq(combined_text, classification_system_prompt)
    
    # Process classification result and get report type
    if isinstance(classification_result, dict):
        report_type = classification_result.get("report_type", "Unknown")
        print(f"Classified as: {report_type}")
        if "confidence_score" in classification_result:
            print(f"Confidence: {classification_result['confidence_score']}")
        if "keywords_identified" in classification_result:
            print(f"Keywords: {classification_result['keywords_identified']}")
    else:
        report_type = "Unknown"
        print("Classification failed, using Unknown as report type")
    
    # Standardize the report type name for file naming
    report_type_standardized = report_type.lower().replace(" ", "_")
    
    # Load the appropriate system prompt based on the classification
    system_prompt = load_system_prompt(report_type_standardized)
    
    # Enhance the extracted text using Groq with all images
    enhanced_text = enhance_text_with_azure(
        extracted_text=combined_text,
        report_type=report_type,
        image_paths=image_paths
    )
    
    return enhanced_text

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
            return {
                "filename": result[0],
                "processed_data": result[1],
                "created_at": result[2],
                "report_type": result[3],
                "patient_name": result[4]
            }
        return None
        
    except Exception as e:
        print(f"Error retrieving report: {e}")
        if 'conn' in locals():
            conn.close()
        return None

# Run the processing
if __name__ == "__main__":
    pdf_path = "Haematology - CBC 13-9-24.pdf"
    process_pdf(pdf_path)