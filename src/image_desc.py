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
                        "text": "You are a professional environmental analyst. Analyze the given image of a water body and generate a concise, structured report addressing the following: 1. Safety Overview: Evaluate whether the water poses immediate health or environmental risks based on visible features like clarity, color, and the presence of potential contaminants. Focus only on what can be seen. 2. Key Features: Briefly identify significant visual aspects such as water clarity, sediment presence, organic material, or surface conditions. Avoid assumptions beyond the visible evidence. 3. Physical Appearance: Describe the water’s appearance (e.g., color, transparency, surface reflection), explaining how these may indicate the water’s general quality without overreaching conclusions. 4. Broad Classification: Provide a general classification (Clean, Polluted, Requires Further Testing) based solely on visible evidence. Clearly justify the classification without excess speculation. 5. Environmental Impact: Discuss how the water's condition may affect surrounding ecosystems or human activity, staying within the scope of the visible evidence and avoiding premature conclusions. 6. Economic Considerations: Mention any potential economic impacts (e.g., treatment needs, impact on local industries), emphasizing that further testing is required to confirm these impacts. 7. Recommendations: Offer limited, relevant recommendations based on visible conditions, without over-prescribing actions. Highlight the need for further investigation if necessary. Ensure the report remains objective and based strictly on what can be observed in the image."
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
