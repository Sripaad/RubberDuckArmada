import gradio as gr
import logging
from PIL import Image
import os

# Import your existing functions
from caller import analyser_api_call, infer_from, create_prognosis_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_image(input_image):
    logger.info("Processing uploaded image")
    
    # Save the uploaded image temporarily
    temp_path = "temp_upload.png"
    input_image.save(temp_path)
    
    # Perform analysis
    infer_from(temp_path)
    prognosis = create_prognosis_report(temp_path)
    result = analyser_api_call(temp_path)
    
    # Clean up temporary file
    os.remove(temp_path)
    
    # Process the result
    if result and 'choices' in result and result['choices']:
        analysis = result['choices'][0]['message']['content']
    else:
        analysis = "Analysis failed. Please try again."
    
    # Return results
    return Image.open("output.jpg"), Image.open("overlay_output.jpg"), prognosis, analysis

# Define the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Segmentation Output"),
        gr.Image(type="pil", label="Overlay Output"),
        gr.Textbox(label="Prognosis Report"),
        gr.Textbox(label="Analysis Result")
    ],
    title="Medical Image Analysis Demo",
    description="Upload a medical image for analysis. The system will provide segmentation, overlay, prognosis, and analysis.",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)