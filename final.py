import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import base64
from groq import Groq
import json
from pdf2image import convert_from_path
import os

# Set up Tesseract path and remove image size limit
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Image.MAX_IMAGE_PIXELS = 500000000

# Initialize Groq client
client = Groq(api_key="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")

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def preprocess_image(image_path):
    """Preprocesses the image to enhance OCR accuracy."""
    image = Image.open(image_path).convert("L")
    image = image.filter(ImageFilter.SHARPEN).resize((3000, 3000), Image.LANCZOS)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2).point(lambda x: 0 if x < 128 else 255, '1')
    return image

def resize_if_necessary(image_path):
    """Resizes the image if it exceeds maximum allowed pixels."""
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

def extract_text_from_image(image_path):
    """Extracts text from a preprocessed image using Tesseract OCR."""
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 --oem 3')
    # print("Extracted text:", text)
    return text

def convert_pdf_to_images(pdf_path):
    """Converts a PDF to images."""
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = f"page_{i}.png"
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

def correct_text_with_groq(text, base64_image):
    """Sends extracted text and image to Groq LLM for correction."""
    prompt = (
        """extract the text from the image"""
    )
    
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }],
        model="llama-3.2-90b-vision-preview",
    )
    
    corrected_text = chat_completion.choices[0].message.content
    # print(f"Corrected text: {corrected_text}")
    return corrected_text


def save_text_to_json(data, filename="result.json"):
    """Saves data in JSON format."""
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data saved to {filename}")

def classify_report_with_groq(corrected_text, system_prompt_content):
    """
    Classifies the report using the Groq LLM.
    
    Args:
        corrected_text (str): The text of the report to classify.
        system_prompt_content (str): The content of the system prompt.

    Returns:
        dict: The classification result in JSON format.
    """
    
    classification_response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
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
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    classification_result = classification_response.choices[0].message.content
    print(f"Classification Result: {classification_result}")
    
    return classification_result


def process_file(file_path, system_prompt_path):
    """
    Processes both image and PDF files and ensures the correct system prompt is passed.
    
    Args:
        file_path (str): Path to the file (image or PDF) to be processed.
        system_prompt_path (str): Path to the system prompt file.
    """
    # Load the system prompt content from file
    try:
        with open(system_prompt_path, 'r') as file:
            system_prompt_content = file.read()
    except FileNotFoundError:
        print(f"Error: System prompt file not found at {system_prompt_path}")
        return
    
    if file_path.lower().endswith('.pdf'):
        print("Processing PDF...")
        image_paths = convert_pdf_to_images(file_path)
        overall_text = ""
        base64_images = []

        # Extract text and encode each page
        for image_path in image_paths:
            base64_images.append(encode_image(image_path))
            extracted_text = extract_text_from_image(image_path)
            overall_text += extracted_text + "\n"  # Combine text from all pages
        
        # Process the combined text
        process_combined_text(overall_text, base64_images, system_prompt_content)
    else:
        print("Processing image...")
        process_image(file_path, system_prompt_content)


def process_combined_text(overall_text, base64_images, system_prompt_content):
    """
    Processes combined text from all pages for correction and classification.
    
    Args:
        overall_text (str): Combined text extracted from all pages.
        base64_images (list): List of base64-encoded images.
        system_prompt_content (str): The content of the system prompt.
    """
    # Correct text with Groq API using the first image as a reference
    corrected_text = correct_text_with_groq( base64_images[0])

    # Classify the report based on the corrected text
    classification_result = classify_report_with_groq(corrected_text, system_prompt_content)

    # Save the corrected text and classification result in JSON format
    save_text_to_json({
        "corrected_text": corrected_text,
        "classification": classification_result
    })


def process_image(image_path, system_prompt_content):
    """
    Processes a single image file (extract text, correct, classify, and save).
    
    Args:
        image_path (str): Path to the image file.
        system_prompt_content (str): The content of the system prompt.
    """
    image_path = resize_if_necessary(image_path)
    
    # Encode image to base64
    base64_image = encode_image(image_path)

    # Extract text from image
    extracted_text = extract_text_from_image(image_path)

    # Correct text with Groq API
    corrected_text = correct_text_with_groq(extracted_text, base64_image)

    # Classify the report with another Groq LLM
    classification_result = classify_report_with_groq(corrected_text, system_prompt_content)

    # Save the corrected text and classification result in JSON format
    save_text_to_json({
        "corrected_text": corrected_text,
        "classification": classification_result
    })


def main():
    file_path = "Haematolog(CBC) (1).pdf"  # Replace with your file path
    system_prompt_path = "classify_sys_prompt.txt"  # Path to the system prompt file
    process_file(file_path, system_prompt_path)


if __name__ == "__main__":
    main()

