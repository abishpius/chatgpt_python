# Enhancing and Ethics
Adding TTS to Gradio is quick.
We can convert our contrastive search results to audio quickly:
```python
  cSearchText = contrastive[0]['generated_text']
  audio = gTTS(text=cSearchText, lang='en', slow=False)
  audio.save("cSearchVoice.wav")
```

Once saved we add the file to our return statement, an audio output to Gradio, and boom! We have enhanced our app in a flash. 

Gradio has many more features and integrations with Natural Language Processing pipelines. The [full documentation](https://gradio.app/docs/) has many blocks you can implement.

 Another great example recently was shared by [João Gante](https://twitter.com/joao_gante/status/1623035369754341377) and highlights the ability to color-code output based on model certainty. 

This is a critical capability because inaccurate information is one of the biggest risks to using large language models along with bias and toxicity.
Meta discussed those risks when they released their paper [Democratizing Access to Large-Scale Language Models](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/), Forbes wrote about [AI Ethics And The Future Of Where Large Language Models Are Heading](https://www.forbes.com/sites/lanceeliot/2022/08/30/ai-ethics-asking-aloud-whether-large-language-models-and-their-bossy-believers-are-taking-ai-down-a-dead-end-path/?sh=2a4df5442250), and that's just the tip of the AI-Ethics rabbit hole that I encourage you to spend time exploring. The more people and perspectives involved in these discussions the better. 