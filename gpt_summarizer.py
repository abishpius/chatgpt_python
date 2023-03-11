import gradio as gr
import os
import openai
import textwrap
import time

openai.api_key = os.getenv("OPENAI_API_KEY")


def save_file(content, filepath):
  with open(filepath, 'w', encoding='utf-8') as outfile:
    outfile.write(content)


def chunk_summarize(alltext, max_length, style):
  # split and convert
  chunks = textwrap.wrap(alltext, 4000)
  result = list()
  count = 0

  for chunk in chunks:
    count = count + 1
    summary = generate_text_gpt(chunk, max_length, style)
    print('\n\n', count, 'of', len(chunks), summary)
    result.append(summary)

  # save a file for reference as well as return full summary
  save_file('\n\n'.join(result), 'outputs/output_%s.txt' % time())

  return '\n\n'.join(result)


def generate_text_gpt(input_string, max_length, style):
  sum_styles = [
    "convert the above content into bullet points",
    "write an executive summary of the above content",
    "shrink the word count of the above content without losing any information"
  ]
  full_prompt = f"Imagine a great writer summarizing the following content while keeping all key information: {input_string} ||END CONTENT|| Inspired by great writers,  {sum_styles[style]} in {max_length*.75} words:"
  response = openai.Completion.create(model="text-davinci-003",
                                      prompt=full_prompt,
                                      temperature=0.1,
                                      max_tokens=max_length,
                                      top_p=1,
                                      frequency_penalty=0,
                                      presence_penalty=0)
  answer = response.choices[0]['text']
  return (answer)


def to_gradio():
  demo = gr.Interface(
    fn=chunk_summarize,
    inputs=[
      gr.Textbox(lines=2, placeholder="Enter content to summarize..."),
      gr.Slider(0, 1000),
      gr.Dropdown(["bullets", "executive", "trim"], type="index")
    ],
    outputs="text")
  demo.launch(debug=True, share=True)


if __name__ == "__main__":
  to_gradio()
