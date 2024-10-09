import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
from groq import Groq
import dspy

load_dotenv()

from src.image_desc import ImageAnalyzer
from src.agent import ReportGenerator

st.set_page_config(page_title="Water Quality Analyzer", layout="wide")

st.title("🌊 Water Quality Analyzer")
st.write("Upload an image of a water body (either from file or capture from your camera) to generate a professional water quality report.")

analyzer = ImageAnalyzer()
report_gen = ReportGenerator()

col1, col2 = st.columns(2)

with col1:
    st.header("📁 Upload an Image")
    image_input = st.file_uploader("Upload a water body image", type=["jpg", "png", "jpeg"])

with col2:
    st.header("📸 Capture from Camera")
    camera_input = st.camera_input("Or capture from your camera")

water_source_label = st.selectbox("Select water source type", ["None", "clean", "river", "sea", "contaminated", "suspected contamination"])

if image_input or camera_input:
    if image_input:
        image = Image.open(image_input)
    else:
        image = Image.open(BytesIO(camera_input.getvalue()))

    with st.expander("Preview Image"):
        st.image(image, caption="Uploaded/Captured Water Body Image", use_column_width=True)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if water_source_label == "None":
        st.warning("Please select a water source type.")
    else:
        # Generate water quality report
        with st.spinner("Generating water quality report..."):
            report_message = analyzer.generate_water_quality_report(
                image_base64=image_base64,
                water_source_label=water_source_label
            )

            formatted_report = report_gen.forward(report_message.content)

        st.markdown("### Generated Water Quality Report")
        st.markdown(formatted_report, unsafe_allow_html=True)

        st.download_button(
            label="Download Report",
            data=formatted_report,
            file_name="water_quality_report.md",
            mime="text/markdown"
        )
