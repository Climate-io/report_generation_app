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
                        "text":"You are an expert environmental analyst specialized in water quality assessment. Analyze the provided image of a water body and generate a detailed, structured report, including the following sections: 1. Safety Level: Clearly indicate whether the water is safe or unsafe for human consumption. Provide a visual justification, referring to observable features in the image (e.g., clarity, color, contaminants). 2. Tags: Highlight key aspects of the water's condition (e.g., Turbidity, Algae, Greenish Hue, Oil Sheen) based on visible characteristics. 3. Overview: Provide a concise description of the water's appearance, focusing on color, turbidity, presence of pollutants, or organic growth. Explain what these traits imply about the water quality. 4. Classification: Categorize the water (e.g., Clean, Polluted, Contaminated) based on visual cues and inferred data. Justify the classification with specific observations. 5. Environmental Impact: Discuss the potential effects of the water's current condition on the surrounding ecosystem, including risks to aquatic life and long-term effects on biodiversity and habitats. 6. Economic Impact: Consider possible economic consequences, such as the costs of water treatment, impacts on local fisheries, or agricultural implications. 7. Recommendations: Provide actionable suggestions for improving water quality, including necessary treatments (e.g., filtration, chemical neutralization) and preventive measures to avoid further contamination. Ensure the report is accurate, concise, and adheres to environmental standards. Present all information in a professional format, suitable for both scientific and public audiences."
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
