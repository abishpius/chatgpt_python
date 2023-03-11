# Gradio Basics

Gradio is a free and open-source Python library that allows you to develop an easy-to-use customizable demo for your machine-learning model that anyone can use anywhere, right in their browser. Gradio integrates with the most popular Python libraries, including Scikit-learn, PyTorch, NumPy, Tensor Flow, and others. 

We can easily get started by checking out their [website and documentation](https://gradio.app/quickstart/). 


Here is an example of how to use Gradio to create an interface in Python:

 ```python
 import gradio as gr

def hello(input):
    
    return "Hello " + input + "!"

demo = gr.Interface(fn=hello, inputs="text", outputs="text")

demo.launch(debug=True, share=True) 
```

How slick is that! We can make a simple interface that works with Python in just a few lines of code. 
