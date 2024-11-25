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
def enhance_text_with_azure(extracted_text, report_type):
    system_prompt, user_prompt = generate_prompt(extracted_text, report_type)
    print(extracted_text)
    # print(f"System Prompt: {system_prompt}")
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}]
        )
        # Get the enhanced text
        enhanced_text = response.choices[0].message.content
        
        # Clean the enhanced text to remove any unnecessary markdown or formatting
        # Look for the JSON-like content within the response and parse it
        try:
            # Remove ```json markdown and any leading/trailing spaces
            json_text = re.sub(r'```json|```', '', enhanced_text).strip()

            
            # Parse the cleaned JSON text
            parsed_json = json.loads(json_text)
            return parsed_json
        except json.JSONDecodeError:
            print("Error parsing the enhanced text as JSON")
            
            return extracted_text  # Fallback to original text if parsing fails
        
    except Exception as e:
        print(f"Error enhancing text with Azure: {e}")
        return extracted_text  # Return the original text if an error occurs

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

def extract_text_from_image(image_path, output_file="extracted_text.txt", preprocessed_image_path="preprocessed_image.png"):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    
    # Extract text from the preprocessed image
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 --oem 3')
    
    # Save the extracted text to a file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(text)
    
    print(f"Text extracted and saved to {output_file}")
    return text

# Function to convert PDF to images using two options with fallback logic
def convert_pdf_to_images_option1(pdf_path):
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        image_paths = []
        for i, image in enumerate(images):
            image_path = f"page_{i}.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        print(f"Option 1 failed with error: {e}")
        return None

def convert_pdf_to_images_option2(pdf_path, scale=300/72, output_folder='.'):
    try:
        pdf_file = pdfium.PdfDocument(pdf_path)
        page_indices = [i for i in range(len(pdf_file))]
        
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=scale,
        )
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_paths = []
        for i, image in zip(page_indices, renderer):
            image_file_path = os.path.join(output_folder, f'{pdf_path[:-4]}_page_{i+1}.jpg')
            image.save(image_file_path, format='JPEG', optimize=True)
            image_paths.append(image_file_path)
        
        return image_paths
    except Exception as e:
        print(f"Option 2 failed with error: {e}")
        return None

def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 --oem 3')
    return text

def convert_pdf_to_images(pdf_path):
    # Attempt Option 1
    image_paths = convert_pdf_to_images_option1(pdf_path)
    if image_paths:
        text = extract_text_from_image(image_paths[0])
        if len(text.split()) >= 10:  # Check if Option 1 extracted 6+ words
            return image_paths
        else:
            print("Option 1 extraction produced less than 10 words. Switching to Option 2.")

    # Fallback to Option 2
    return convert_pdf_to_images_option2(pdf_path)

def save_text_to_json(extracted_text, report_type):
    # Create a directory for the report type if it doesn't exist
    folder_path = os.path.join("extracted_reports", report_type)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Create a unique filename using the current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder_path, f"extracted_text_{timestamp}.json")
    
    # Save the enhanced text (which is now a parsed JSON object)
    with open(filename, "w") as file:
        json.dump(extracted_text, file, indent=4)
    
    print(f"Enhanced text saved to {filename}")

    
def classify_medical_report(report_text):
    report_text_lower = report_text.lower()

    # Helper function to calculate matches
    def calculate_matches(keywords):
        matches = [keyword for keyword in keywords if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', report_text_lower)]
        return matches, len(matches), len(keywords)

    # Count matches and prepare details
    cbc_matches, cbc_count, total_cbc = calculate_matches(cbc_keywords)
    serum_matches, serum_count, total_serum = calculate_matches(serum_keywords)
    endoscopy_matches, endoscopy_count, total_endoscopy = calculate_matches(endoscopy_keywords)
    clinical_biochemistry_matches, clinical_biochemistry_count, total_clinical = calculate_matches(clinical_biochemistry_keywords)

    # Calculate match percentages
    match_percentages = {
        "CBC": (cbc_count / total_cbc) * 100 if total_cbc > 0 else 0,
        "Serum": (serum_count / total_serum) * 100 if total_serum > 0 else 0,
        "Endoscopy": (endoscopy_count / total_endoscopy) * 100 if total_endoscopy > 0 else 0,
        "Clinical Biochemistry": (clinical_biochemistry_count / total_clinical) * 100 if total_clinical > 0 else 0,
    }

    # Determine classification based on highest percentage
    report_type = max(match_percentages, key=match_percentages.get)

    # Create classification details for printing
    classification_details = {
        "report_type": report_type,
        "match_details": {
            "CBC": {"matched_keywords": cbc_matches, "match_percentage": match_percentages["CBC"]},
            "Serum": {"matched_keywords": serum_matches, "match_percentage": match_percentages["Serum"]},
            "Endoscopy": {"matched_keywords": endoscopy_matches, "match_percentage": match_percentages["Endoscopy"]},
            "Clinical Biochemistry": {"matched_keywords": clinical_biochemistry_matches, "match_percentage": match_percentages["Clinical Biochemistry"]},
        }
    }
    print(classification_details)
    return classification_details

pdf_path = "Haematology - CBC 26-9-24.pdf"
image_paths = convert_pdf_to_images(pdf_path)

# Let's extract text from the first image
extracted_text = extract_text_from_image(image_paths[0])

# Classify the extracted text
classification = classify_medical_report(extracted_text)
report_type = classification["report_type"]

# Enhance the extracted text using Azure OpenAI
enhanced_text = enhance_text_with_azure(extracted_text, report_type)

# Save the enhanced text under the corresponding report type folder
save_text_to_json(enhanced_text, report_type)
