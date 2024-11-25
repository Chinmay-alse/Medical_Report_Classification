from groq import Groq
import base64
from PIL import Image



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "CBC - Complete Blood Count_page_1.jpg"
img = Image.open(image_path)

# Get the current dimensions
width, height = img.size
current_pixels = width * height

# Define the maximum number of pixels
max_pixels = 33177600

# If the image exceeds the maximum size, resize it
if current_pixels > max_pixels:
    scale_factor = (max_pixels / current_pixels) ** 0.3
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image
    img_resized = img.resize((new_width, new_height))

    # Save the resized image
    img_resized.save("resized_image.png")
    print(f"Image resized to {new_width}x{new_height}")
    image_path = 'resized_image.png'
else:
    print("Image size is within the allowed limit.")
    


# Getting the base64 string
base64_image = encode_image(image_path)

client = Groq(api_key="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")

extracted_text = '''
; an 4
Laboratory Medicine Ca ity
Unit of Narayana Health
Department of LABORATORY MEDICINE
Lab Report : HAEMATOLOGY LAB
MRN 10110000099422 Department OBSTETRICS & GYNAECOLOGY
Name Mrs JONAKI MAHAJAN Specimen WHOLE BLOOD
Age / Sex 45 Year(s) 1 Month(s/FEMALE Visit Type OP / OP-001
/ Mobile No 9051222953
Sample No 021910050383 Collected On 05/10/2019 09:58
.° Consulting Doctor Dr Manjula H M Referring Doctor Dr Manjula HM
Patient Address PURVA SKYWOOD, HARALUR, Received On 05/10/2019 10:31
Bangalore, Karnataka Reported On 05/10/2019 12:06
‘". ‘Tést Name Result ©-. i}, Unit "| | Biological Reference
COMPLETE BLOOD COUNT(CBC) es EE :
I. MOGLOBIN (HB%) | g/dL 12.0-15.0
(Photometric Measurement) . i
RED BLOOD CELL COUNT :3.99 -Million/uL '3.8-4.8
(Electrical Impedance) : \ :
PCV (PACKED CELL VOLUME) / 26.7 L i% : 36.0-46.0
HEMATOCRIT :
(Calculated) . 7 | 7 .
MCV (MEAN CORPUSCULAR VOLUME) 66.8 L ‘fl *83.0-101.0
(Derived) :
MCH (MEAN CORPUSCULAR 20.8 L : Pg :27.0-32.0 :
HAEMOGLOBIN)
MCHC (MEAN CORPUSCULAR 31.1 L Te -31.5-34.5
HAEMOGLOBIN CONCENTRATION) :
REQ CELL DISTRIBUTION WIDTH 18.7 '% 11.6-14.0
“rL@TELET COUNT 284 ‘ Thous/uL _ 150.0-450.0
(E'-rical Impedance plus Microscopy) : : .
Tay AL LEUKOCYTE COUNT (TLC) 5.7 : Thous/pL :4.0-10.0
(EleBtrical Impedance)
& |
v
'''

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Add whatever information is missing to {extracted_text} based on the textual and numerical data given in the image. I only want the numerical values to be added and nothing else. The rest of the extracted text should remain same."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",
)

print(chat_completion.choices[0].message.content)