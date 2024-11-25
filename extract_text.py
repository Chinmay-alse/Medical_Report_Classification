import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from groq import Groq
import base64
from io import BytesIO
import re

Image.MAX_IMAGE_PIXELS = 500000000

def preprocess_image(image_path):
    """
    Preprocesses the image and returns its data URL.
    """
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((3000, 3000), Image.LANCZOS)
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    image = image.point(lambda x: 0 if x < 128 else 255, '1')
    
    # Convert the preprocessed image to data URL
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    image_data_url = f"data:image/png;base64,{img_base64}"
    
    return image_data_url

def extract_text_with_groq(image_data_url):
    """
    Sends the preprocessed image to Groq's vision model for text extraction.
    """
    client = Groq(api_key="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")
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extract all text from this medical report image with these requirements:
                        1. Extract all text exactly as it appears
                        2. Maintain original formatting and line breaks
                        3. Keep all numbers, units, and special characters exactly as shown
                        4. Do not add any interpretations or additional formatting
                        5. Present the text in a clean, readable format"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }],
            temperature=0.1,  # Lower temperature for more exact extraction
            max_tokens=1024,
        )
        
        if completion.choices and len(completion.choices) > 0:
            extracted_text = completion.choices[0].message.content
            # Clean the response
            try:
                cleaned_text = re.sub(r'```json\s*|\s*```', '', extracted_text).strip()
                # Preserve newlines and spaces as they are
                cleaned_text = cleaned_text.replace("\n", "\n")  # Make sure the newlines are intact
                
                return cleaned_text
            except Exception as e:
                return f"Error cleaning text: {str(e)}"
        else:
            return "No text extracted."
    except Exception as e:
        return str(e)


def process_with_groq_system_prompt(extracted_text, system_prompt):
    """
    Sends the extracted text and system prompt to Groq's text model.
    """
    client = Groq(api_key="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")
    
    try:
        # Send the extracted text and system prompt to the Groq model
        completion = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt  # Custom system prompt
                },
                {
                    "role": "user",
                    "content": extracted_text  # Text extracted from the image
                }
            ],
            temperature=0.7,  # Adjust temperature as needed
            max_tokens=1024,
            top_p=1,
            stream=True,  # You can use streaming for real-time processing
        )

        # Collect and return the response from the assistant
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
            return chunk.choices[0].delta.content  # You can collect and store the response if needed

    except Exception as e:
        print(f"Error during Groq API call: {str(e)}")
        return None

def main():
    input_path = "CBC - Complete Blood Count_page_1.jpg"
    system_prompt_path = "cbc_prompt.txt"  # Adjust this to the correct path of your system prompt file
    results = {}

    # Read the system prompt from the file
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    else:
        print(f"System prompt file not found at {system_prompt_path}.")
        return

    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_data_url = preprocess_image(file_path)
                extracted_text = extract_text_with_groq(image_data_url)
                
                if extracted_text:
                    processed_output = process_with_system_prompt(extracted_text, system_prompt)
                    results[filename] = {
                        "extracted_text": extracted_text,
                        "processed_output": processed_output
                    }

    # Output the result as a JSON file
    with open("processed_results.json", 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
