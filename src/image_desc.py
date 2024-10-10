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

    def generate_water_quality_report(self, image_base64, water_source_label, temperature=1, max_tokens=1024, top_p=1):
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
                        "text": "You are a professional environmental analyst. Generate a structured report based on the given image of a water body, covering the following: Safety Level: Indicate if the water is safe or unsafe for consumption with visual justification. Tags: Summarize key aspects of the water's condition (e.g., Turbidity, Algae, Greenish Hue). Overview: Briefly describe the water's appearance (color, turbidity, pollutants) and what it implies about quality. Classification: Classify the water (e.g., Clean, Polluted) and explain the reasoning. Environmental Impact: Discuss potential risks to the ecosystem, aquatic life, and human health. Economic Impact: Note possible economic effects (e.g., treatment costs, fisheries). Recommendations: Suggest treatments, preventive measures, and improvements for water quality. Ensure the report is concise, professional, and aligned with environmental standards."

                    },
                    {
                        "type": "text",
                        "text": f"Analyze this water body labeled as {water_source_label} and provide a detailed environmental report."
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
    analyzer = ImageAnalyzer(model_name="llava-v1.5-7b-4096-preview")
    # image_base64 = analyzer.encode_image("./test_data/2.jpg")
    report_message = analyzer.generate_water_quality_report(
        image_path="./test_data/2.jpg",
        water_source_label="contaminated"
    )
    print(report_message)
