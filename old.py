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
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

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
def enhance_text_with_groq(extracted_text, report_type):
    """
    Enhances the extracted text using the Groq LLM based on the report type.
    """
    system_prompt, user_prompt = generate_prompt(extracted_text, report_type)
    print(f"Using system prompt for: {report_type}")
    
    try:
        response = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",  # Make sure this is set to your Groq model
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.7,
            max_tokens=2024,
            top_p=1
        )
        
        # Get the enhanced text
        enhanced_text = response.choices[0].message.content
        # print("Raw Enhanced Text:", enhanced_text)  # Debug print
        
        return enhanced_text
            
    except Exception as e:
        print(f"Error in API call: {e}")
        return None
        
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

def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 --oem 3')
    print(text)
    return text

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
            for i, img_path in enumerate(image_paths):
                text = extract_text_from_image(img_path)
                if len(text.split()) >= 10:
                    # Move the image to the final location
                    final_path = os.path.join(output_dir, f"{base_name}_page_{i+1}_converted.png")
                    if os.path.exists(img_path):
                        os.replace(img_path, final_path)
                        final_image_paths.append(final_path)
                if os.path.exists(img_path):
                    os.remove(img_path)
            
            if final_image_paths:
                return final_image_paths
            
            print("Option 1 extraction produced insufficient text. Switching to Option 2.")
    except Exception as e:
        print(f"Error in Option 1: {e}")
    
    # Fallback to Option 2
    try:
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

def save_text_to_json(extracted_text, report_type):
    if not extracted_text:
        print("No text to save")
        return
    
    # Create a directory for the report type if it doesn't exist
    folder_path = os.path.join("extracted_reports", report_type)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Create a unique filename using the current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder_path, f"extracted_text_{timestamp}.json")
    
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
    
def check_missing_parameters(enhanced_text, report_type):
    """
    Check if all required parameters are present in the enhanced text.
    Returns a list of missing parameters.
    """
    try:
        # Parse the enhanced text if it's a string
        if isinstance(enhanced_text, str):
            if enhanced_text.startswith('```json'):
                enhanced_text = enhanced_text[7:-3]
            parsed_text = json.loads(enhanced_text)
        else:
            parsed_text = enhanced_text

        # Define required parameters based on report type
        required_parameters = {
            "CBC": [
                "Hemoglobin",
                "Total WBC/leukocyte count",
                "Platelet count",
                "Hematocrit/PCV",
                "Total RBC count",
                "WBC/Leukocyte differential count",
                "RBC indices"
            ],
            "Serum":[
                "CA15-3/CA27.29",
                "CA19-9",
                "CA125",
                "CEA"
            ]
            # Add other report types as needed
        }

        # Get the parameters from the enhanced text
        found_parameters = set()
        for param in parsed_text.get('parameters', []):
            found_parameters.add(param.get('parameter'))

        # Check which required parameters are missing
        missing_parameters = []
        for required_param in required_parameters.get(report_type, []):
            if required_param not in found_parameters:
                missing_parameters.append(required_param)

        return missing_parameters

    except Exception as e:
        print(f"Error checking parameters: {e}")
        return None
    
def enhance_text_with_vision(extracted_text, report_type, image_paths):
    """
    Enhances the text using the vision model with support for multiple images.
    """
    system_prompt, user_prompt = generate_prompt(extracted_text, report_type)
    print(f"Using vision model for: {report_type}")
    
    try:
        # Prepare the content with both text and images
        content = [
            {
                "type": "text",
                "text": f"This is a {len(image_paths)}-page medical report. Add whatever information is missing to {extracted_text} based on the textual and numerical data given in all the images. Combine all information from all pages into a single comprehensive output. I only want the numerical values to be added and nothing else. The rest of the extracted text should remain same."
            }
        ]
        
        # Add all images to the content
        for i, image_path in enumerate(image_paths):
            base64_image = encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            })
            print(f"Added image {i+1}/{len(image_paths)} to vision request")
        
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0.7,
            max_tokens=2024,
            top_p=1
        )
        
        return response.choices[0].message.content
            
    except Exception as e:
        print(f"Error in vision API call: {e}")
        return None
    
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


# At the start of processing
start_time = time.time()

pdf_path = "Haematology - CBC 26-9-24.pdf"
image_paths = convert_pdf_to_images(pdf_path)

if not image_paths:
    print("Failed to convert PDF to images")
    exit()

# Extract text from all images and combine
combined_text = ""
for i, image_path in enumerate(image_paths):
    page_text = extract_text_from_image(image_path)
    combined_text += f"\nPage {i+1}:\n{page_text}"

print(f"Extracted text from {len(image_paths)} pages")

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

# Enhance the combined text using Groq
enhanced_text = enhance_text_with_groq(combined_text, report_type)

# After getting enhanced_text from enhance_text_with_groq
if enhanced_text:
    # Check for missing parameters
    missing_parameters = check_missing_parameters(enhanced_text, report_type)
    
    if missing_parameters:
        print(f"Missing parameters detected: {missing_parameters}")
        print("Attempting to extract missing information using vision model...")
        
        # Use vision model to get missing parameters
        vision_enhanced_text = enhance_text_with_vision(enhanced_text, report_type, image_paths)
        
        if vision_enhanced_text:
            # Check if vision model found the missing parameters
            vision_missing_params = check_missing_parameters(vision_enhanced_text, report_type)
            
            if not vision_missing_params:
                print("Successfully extracted all parameters using vision model")
                enhanced_text = vision_enhanced_text
            else:
                print(f"Still missing parameters after vision processing: {vision_missing_params}")
    
    # Save the final enhanced text
    save_text_to_json(enhanced_text, report_type)
    
    # Process for logging
    try:
        parsed_json = json.loads(enhanced_text) if isinstance(enhanced_text, str) else enhanced_text
        enhanced_text_for_log = parsed_json
    except json.JSONDecodeError:
        enhanced_text_for_log = {"raw_text": enhanced_text}
    
    # Document the process
    document_report_processing(
        pdf_path=pdf_path,
        classification_result=classification_result,
        enhanced_text=enhanced_text_for_log,
        processing_time=time.time() - start_time
    )
else:
    print("No enhanced text was generated")