import streamlit as st
import base64
from PIL import Image
import io
import pandas as pd
import os
import subprocess
import sys


# Check and install mistralai if not installed
try:
    from mistralai import Mistral
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mistralai"])
    from mistralai import Mistral  # Import again after installation

from dotenv import load_dotenv

load_dotenv()

def encode_image_to_base64(image_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

def parse_ai_response(response_text):
    """Parse the AI response into a structured format"""
    results = {}
    lines = response_text.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            value = value.split(' ')[0] if ' ' in value else value
            results[key] = value
    return results

def analyze_cylinder_image(image_bytes):
    api_key = os.getenv("MISTRAL_API_KEY")
    model = "pixtral-12b-2409"
    client = Mistral(api_key=api_key)
    base64_image = encode_image_to_base64(image_bytes)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the engineering drawing and provide the following information with these strict rules 1)in exactly this format, 2)only extract these fields from the upload image rest keep empty in the table(if empty then return NA):3)if the metrics dont match ,convert them into the desired metrics 4)Try your best to get all the values ,as this is a steel pipe image\n"
                        "CYLINDER ACTION: [value]\n"
                        "BORE DIAMETER: [value] MM\n"
                        "OUTSIDE DIAMETER: [value] MM\n"
                        "ROD DIAMETER: [value] MM\n"
                        "STROKE LENGTH: [value] MM\n"
                        "CLOSE LENGTH: [value] MM\n"
                        "OPEN LENGTH: [value] MM\n"
                        "OPERATING PRESSURE: [value] BAR\n"
                        "OPERATING TEMPERATURE: [value] DEG C\n"
                        "MOUNTING: [value]\n"
                        "ROD END: [value]\n"
                        "FLUID: [value]"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": base64_image
                }
            ]
        }
    ]
    
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return None

def main():
    # Set page config
    st.set_page_config(
        page_title="Technical Drawing DataSheet Extraction",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            border-radius: 20px;
            padding: 10px 24px;
            background-color: #0083B8;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0073A8;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("Technical Drawing DataSheet Extraction")

    # Define parameters and units
    parameters = {
        "CYLINDER ACTION": "N/A",
        "BORE DIAMETER": "MM",
        "OUTSIDE DIAMETER": "N/A",
        "ROD DIAMETER": "MM",
        "STROKE LENGTH": "MM",
        "CLOSE LENGTH": "MM",
        "OPEN LENGTH": "N/A",
        "OPERATING PRESSURE": "BAR",
        "OPERATING TEMPERATURE": "DEG C",
        "MOUNTING": "N/A",
        "ROD END": "N/A",
        "FLUID": "N/A"
    }

    # File uploader and processing section
    uploaded_file = st.file_uploader("Select File", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if 'results_df' not in st.session_state:
                st.session_state.results_df = None

            # Process button
            if st.button("Process Drawing", key="process_button"):
                with st.spinner('Processing drawing...'):
                    # Get image bytes for processing
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    # Process the image
                    result = analyze_cylinder_image(image_bytes)
                    
                    if result:
                        # Parse the AI response
                        parsed_results = parse_ai_response(result)
                        
                        # Create DataFrame for display
                        st.session_state.results_df = pd.DataFrame([
                            {"Parameter": k, "Value": v, "UOM": parameters[k]}
                            for k, v in parsed_results.items()
                            if k in parameters
                        ])
                        st.success("Drawing processed successfully!")

            # Display the table if results exist
            if st.session_state.results_df is not None:
                st.write("### Extracted Parameters")
                st.table(st.session_state.results_df)
            
                # Export button
                csv = st.session_state.results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="cylinder_parameters.csv",
                    mime="text/csv"
                )

        with col2:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Technical Drawing")

if __name__ == "__main__":
    main()
