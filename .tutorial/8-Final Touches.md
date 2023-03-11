# Final Touches

Today in Day 2 we: 
- Learned what Gradio is and how to use it to create a user interface
- Learned how to use Replit Secrets to store API keys in a secure way
- Built a user interface in Gradio for our text generation demo
- Enhanced the user interface with Text-to-Speech
- Thought about the ethical considerations of using large language models

Somethings to think about before joining us for Day 3:
- What other types of large language models are there?
- How can we use language models to solve problems like text summarization?
- What other tasks can these models be used for?

Continue through this short course to learn more!

## Checkout AI Camp!
<img src="https://i.imgur.com/cm5IS8V.png" width="100px" height="100px" id="ai-camp">

 **If you are a teenager interested in joining a community solving problems with technology and exploring careers in tech, check out AI Camp!**

  We offer 1-week and 3-week camps to start, but exceptional students can join our [Team Tomorrow](https://teamtomorrow.com/). **_TT members work after school + weekends on paid internal and external projects_**, opening doors for their future. 

In fact TT member Will Crouch, a High School Junior, helped create todays content!


You can find a copy of the working code below:
``` python
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
```