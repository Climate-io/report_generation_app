import os
import dspy
from dotenv import load_dotenv
from dspy import InputField, OutputField, Signature, Module

load_dotenv()

lm = dspy.OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), max_tokens=4096)

dspy.settings.configure(lm=lm)

class AnalyzeAndFixVLMOutput(Signature):
    """
    Analyze and correct the output from the Vision Language Model (VLM).
    ---
    Input:
    - VLM output: image classification, environmental impact, water quality report.
    
    Output:
    - A professional, fact-checked report free from duplicated sentences, dates, locations, or other real-world identifiers. 
      Ensures content sounds natural, accurate, and provides actionable insights.
    """
    vlm_output = InputField(type=str, desc="VLM output: image classification, environmental impact, water quality report.")
    corrected_report = OutputField(type=str, desc="Fact-checked and corrected report with real-world information like dates, places, and locations removed. Ensures no duplicated sentences.")

class TextToMarkdown(Signature):
    """
    Convert the report text to professional Markdown format.
    ---
    Input:
    - Text report
    
    Output:
    - A well-structured Markdown report with clear headings, bullet points, and proper formatting to enhance readability.
    """
    report = InputField(type= str, desc="Text report.")
    markdown = OutputField(type=str,desc="Professionally formatted Markdown report with clear headings and structure.")
    
class English2Urdu(Signature):
    """
    Translate the provided English text into professional and accurate Urdu.
    ---
    Input:
    - English text
    
    Output:
    - Urdu translation of the text, maintaining the original meaning, tone, and professionalism.
    """
    english_text = InputField(type=str, desc="English text to be translated.")
    urdu_text = OutputField(type=str, desc="Urdu translation of the text with accurate meaning and tone.")

class ReportGenerator(Module):
    def __init__(self):
        super().__init__()
        self.parser = dspy.ChainOfThought(AnalyzeAndFixVLMOutput)
        self.converter = dspy.ChainOfThought(TextToMarkdown)
        self.urdu_converter = dspy.ChainOfThought(English2Urdu)

    def forward(self, vlm_output):
        corrected_report = self.parser(vlm_output=str(vlm_output))
        urdu_report = self.urdu_converter(english_text=str(corrected_report.corrected_report))
        
        markdown = self.converter(report=str(corrected_report.corrected_report))
        markdown_urdu = self.converter(report=str(urdu_report.urdu_text))
        
        return markdown.markdown, markdown_urdu.markdown

