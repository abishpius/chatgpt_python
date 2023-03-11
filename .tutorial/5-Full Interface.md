# Text Generation Interface
We are all set up, let's make an interface for our text generation app. 

Huggingface actually acquired Gradio because it really provides a lot of value so no surprise, there is [great documentation](https://gradio.app/using-hugging-face-integrations/) on how to integrate the two. 

**Using the completed code from Day 1 as reference, try getting it integrated with Gradio on your own:**
```Python 
API_Token = ""

from huggingface_hub.inference_api import InferenceApi

inference = InferenceApi(repo_id="gpt2-large", token=API_Token)

# defining our parameters
top_p_params = {"max_length": 128, "top_p": 0.95, "do_sample": True}
top_k_params = {"max_length": 128, "top_k": 4, "do_sample": True}
contrastive_params = {"max_length": 128, "penalty_alpha": 0.6, "top_k": 4}

input_string = "Learn about AI because"
result = inference(input_string)
print(result)

result = inference(input_string, top_p_params)
print("Top P: {}".format(result))

result = inference(input_string, top_k_params)
print("Top K: {}".format(result))

result = inference(input_string, contrastive_params)
print("Contrastive: {}".format(result))
```

Don't forget to import Gradio!
```python
import gradio as gr
```

