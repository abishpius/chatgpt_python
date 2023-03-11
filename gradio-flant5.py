from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def generate_text_flan(input_string, max_length):
  inputs = tokenizer(input_string, return_tensors="pt")
  outputs = model.generate(**inputs, max_length=max_length)
  final_text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)

  return (final_text)

def to_gradio():
  demo = gr.Interface(fn=generate_text_flan,
                      inputs=["text", gr.Slider(0, 250)],
                      outputs="text")
  demo.launch(debug=True, share=True)


if __name__ == "__main__":
  to_gradio()
