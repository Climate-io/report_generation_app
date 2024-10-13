import base64
from dotenv import load_dotenv
from groq import Groq
import os
# Load environment variables
load_dotenv()

class ImageAnalyzer:
    def __init__(self, model_name="llama-3.2-11b-vision-preview"):
        # self.image_path = image_path
        self.model_name = model_name
        self.client = Groq(timeout=120, api_key=os.getenv("GROQ_API_KEY"))

    def encode_image(self, image_path):
        """
        Encodes the image from the given file path into base64 format.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_water_quality_report(self, image_base64, temperature=1, max_tokens=2048, top_p=1):
        """
        Generates a comprehensive water quality report based on the provided water source image and label.
        """
        # image_base64 = self.encode_image(image_path=image_path)
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a professional environmental analyst. Generate a structured report based on the given image of a water body, covering the following: Safety Assessment: Provide a general assessment of whether the water may pose risks to health or the environment. Justify your assessment based on visible features such as clarity, color, or presence of contaminants. Key Features: Identify and summarize noticeable visual aspects of the water's condition (e.g., Clarity, Sediment, Surface Reflection, Organic Material). Appearance Overview: Briefly describe the water's physical appearance (color, transparency, surface elements) and what it could imply about water quality, without jumping to conclusions. Preliminary Classification: Offer a broad classification (e.g., Clean, Potentially Polluted, Requires Further Analysis) with an explanation for your reasoning, considering both natural and human-influenced factors. Potential Environmental Impact: Discuss how the observed condition might affect the surrounding ecosystem, aquatic life, or human activities, while remaining open-ended about unknowns. Economic Considerations: Highlight possible economic implications based on visible conditions, such as increased treatment needs or impact on fisheries, but emphasize the need for further testing to confirm. Recommendations: Provide broad suggestions for water quality management, including potential actions or preventive measures to maintain or improve water quality. Ensure the report remains professional, neutral, and based on visible evidence, acknowledging areas that may require further investigation."
                    },
                    {
                        "type": "text",
                        "text": f"Analyze this water body and provide a detailed environmental report."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
            stop=None
        )

        return completion.choices[0].message

if __name__ == "__main__":
    # analyzer = ImageAnalyzer(model_name="llava-v1.5-7b-4096-preview")
    analyzer = ImageAnalyzer()
    # image_base64 = analyzer.encode_image("./test_data/2.jpg")
    report_message = analyzer.generate_water_quality_report(
        image_path="./test_data/2.jpg",
        water_source_label="contaminated"
    )
    print(report_message)
