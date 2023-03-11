import gradio as gr
import os
from huggingface_hub.inference_api import InferenceApi
from gtts import gTTS

API_Token = os.environ['HFKEY']
inference = InferenceApi(repo_id="gpt2-large", token=API_Token)


def text_gen(input_string, max_length):
  # defining our parameters
  top_p_params = {"max_length": max_length, "top_p": 0.95, "do_sample": True}
  top_k_params = {"max_length": max_length, "top_k": 4, "do_sample": True}
  contrastive_params = {
    "max_length": max_length,
    "penalty_alpha": 0.6,
    "top_k": 4
  }
  deterministic = inference(input_string)
  top_p = inference(input_string, top_p_params)
  top_k = inference(input_string, top_k_params)
  contrastive = inference(input_string, contrastive_params)
  cSearchText = contrastive[0]['generated_text']
  audio = gTTS(text=cSearchText, lang='en', slow=False)
  audio.save("cSearchVoice.wav")

  return deterministic, top_p, top_k, contrastive, "cSearchVoice.wav"


def to_gradio():

  demo = gr.Interface(
    fn=text_gen,
    inputs=["text", gr.Slider(0, 250)],
    outputs=["text", "text", "text", "text", "audio"],
    title="Text Generation with GPT-2 Language Model",
    description=
    "This model takes some input text, and generates new text from GPT-2.")
  demo.launch(debug=True, share=True)


if __name__ == "__main__":
  to_gradio()
