import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
import dspy
load_dotenv()
from src.image_desc import ImageAnalyzer
from src.agent import ReportGenerator

st.set_page_config(
    page_title="Water Quality Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #0083B8;
        color: white;
    }
    .reportSection {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #0083B8;
    }
    .stSelectbox {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

analyzer = ImageAnalyzer()
report_gen = ReportGenerator()

with st.sidebar:
    st.image(f"./black-logo.png", width=200)
    st.title("Navigation")
    st.markdown("### How to use")
    st.info("""
    1. Upload or capture an image
    2. Select water source type
    3. Wait for analysis
    4. Download the report
    """)
    st.markdown("### About")
    st.write("This app, part of the Climate.io project, analyzes water quality from images using advanced AI techniques. Climate.io aims to revolutionize environmental monitoring by utilizing visual language models and real-time sensor data for water quality assessment.")


tab1, tab2 = st.tabs(["📊 Analysis", "ℹ️ Help"])

with tab1:
    st.title("🌊 Water Quality Analyzer")
    st.markdown("""
        <div style='background-color: #f0f2f6; border-radius: 10px;'>
        Upload an image of a water body (either from file or capture from your camera) 
        to generate a professional water quality report.
        </div>
    """, unsafe_allow_html=True)

    
    st.markdown("### 📁 Upload an Image")
    image_input = st.file_uploader(
        "Drop your file here",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )

    st.markdown("### 📸 Capture from Camera")
    if st.button("Capture from Camera"):
        camera_input = st.camera_input("Click to capture", help="Make sure you have good lighting")
    else:
        camera_input = None

    st.markdown("### 🏷️ Water Source Classification")
    water_source_label = st.selectbox(
        "Select water source type",
        ["None", "clean", "river", "sea", "contaminated", "suspected contamination"],
        help="Choose the type of water body in the image"
    )

    if image_input or camera_input:
        with st.spinner("Processing image..."):
            if image_input:
                image = Image.open(image_input)
            else:
                image = Image.open(BytesIO(camera_input.getvalue()))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            with st.expander("🖼️ Preview Image", expanded=False):
                st.image(image, caption="Uploaded/Captured Water Body Image", use_column_width=True)

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            if water_source_label == "None":
                st.warning("⚠️ Please select a water source type to continue.")
            else:
                with st.spinner("🔄 Analyzing water quality..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    report_message = analyzer.generate_water_quality_report(
                        image_base64=image_base64,
                        water_source_label=water_source_label
                    )
                    formatted_report = report_gen.forward(report_message.content)

                st.markdown("""
                    <div class='reportSection'>
                    <h3 style='color: #0083B8;'>📑 Water Quality Report</h3>
                    """, unsafe_allow_html=True)
                
                formatted_report = formatted_report.replace("```", "")
                formatted_report = formatted_report.replace("markdown", "")
                st.markdown(formatted_report, unsafe_allow_html=True)
                
                if st.download_button(
                    label="📥 Download Report",
                    data=formatted_report,
                    file_name="water_quality_report.md",
                    mime="text/markdown"
                ):
                    st.success("✅ Report downloaded successfully!")
                
                st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("Help & Documentation")
    st.markdown("""
    ### Frequently Asked Questions

    <details>
    <summary>📷 What types of images work best for water quality analysis?</summary>
    To ensure the highest accuracy in analysis, we recommend using clear, high-resolution images of water bodies. Ideally, the images should be well-lit, taken in natural daylight, and free from obstructions such as reflections, shadows, or debris that might obscure the water's surface.
    </details>

    <details>
    <summary>⏱️ How long does the analysis process take?</summary>
    The analysis typically takes between 30 and 60 seconds. However, this may vary based on the size and quality of the image, as well as the complexity of the water body in the picture. Larger images or more complex scenes might take slightly longer.
    </details>

    <details>
    <summary>📄 In what format is the water quality report generated?</summary>
    The water quality report is generated in Markdown format. This allows for easy readability and seamless conversion to other formats like PDF, HTML, or DOCX using external tools, making it versatile for sharing and presentation.
    </details>

    <details>
    <summary>🌍 Can this tool analyze any type of water body?</summary>
    Yes, the app is designed to analyze various types of water bodies such as rivers, seas, lakes, and even suspected contaminated sources. The tool uses advanced AI models tailored to detect the quality of different water sources based on their unique visual characteristics.
    </details>

    <details>
    <summary>📥 How can I download the water quality report?</summary>
    After the analysis is complete, you will be given the option to download the generated report directly from the interface by clicking the "Download Report" button. The report will be saved in Markdown format for your convenience.
    </details>
    """, unsafe_allow_html=True)
