import pypdfium2 as pdfium
import os
import base64
# from groq import Groq

class pdf_convert:

    def convert_pdf_to_images(file_path, scale=300/72, output_folder='.'):
        pdf_file = pdfium.PdfDocument(file_path)  
        page_indices = [i for i in range(len(pdf_file))]
        
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices, 
            scale=scale,
        )
        
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_paths = []  # List to store paths to saved images
        
        for i, image in zip(page_indices, renderer):
            image_file_path = os.path.join(output_folder, f'{file_path[:-4]}_page_{i+1}.jpg')
            image.save(image_file_path, format='JPEG', optimize=True)
            image_paths.append(image_file_path)
        
        return image_paths

    # def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
pdf_path = 'CBC - Complete Blood Count.pdf'
output_folder = r'C:\Users\chinmay alse\Desktop\text extraction'

image_paths = pdf_convert.convert_pdf_to_images(pdf_path, output_folder=output_folder)
# print(image_paths)