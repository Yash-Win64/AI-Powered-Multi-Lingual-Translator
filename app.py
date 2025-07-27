import gradio as gr

def translate(text, language):
    # Dummy translation logic — replace with your own
    return f"Translated '{text}' to {language}"

iface = gr.Interface(
    fn=translate,
    inputs=["text", "text"],
    outputs="text",
    title="Multilingual Translator"
)

iface.launch()
