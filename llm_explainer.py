import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyBy3kt59wkbMeP-6fCLN-DY-6pmJ6lni8o")

def explain_with_llm(
    label: str,
    confidence: float,
    xai_text: str
) -> str:

    prompt = f"""
You are an AI safety assistant.
You are ONLY allowed to explain what is explicitly stated below.
Do NOT speculate or add visual details.

MODEL OUTPUT:
- Prediction: {label}
- Confidence: {confidence:.2f}

XAI SUMMARY:
{xai_text}

TASK:
Explain why the model made this prediction using ONLY the information above.
If something is unclear, state that clearly.
"""

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0
        )
    )


    return response.text

