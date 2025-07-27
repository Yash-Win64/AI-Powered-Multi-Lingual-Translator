import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh",
    "Russian": "ru",
    "Japanese": "ja",
}

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = LANGUAGE_CODES[src_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODES[tgt_lang]])
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("<h1 style='text-align: center; color: #A78BFA;'>🔵 AI-Powered Multi-Lingual Translator</h1>")
    gr.Markdown("<p style='text-align: center;'>✨ Powered by M2M100 - Translate between 8 languages instantly ✨</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("📝 **Input Text**")
            input_text = gr.Textbox(placeholder="Enter the text you want to translate...", lines=4)
        with gr.Column():
            gr.Markdown("📤 **Translation Output**")
            output_text = gr.Textbox(placeholder="Translation will appear here...", lines=4)

    with gr.Row():
        src_lang = gr.Dropdown(choices=list(LANGUAGE_CODES.keys()), label="🌐 Source Language", value="English")
        tgt_lang = gr.Dropdown(choices=list(LANGUAGE_CODES.keys()), label="🎯 Target Language", value="Hindi")

    translate_btn = gr.Button("🚀 Translate Now", elem_classes=["translate-btn"])

    translate_btn.click(fn=translate, inputs=[input_text, src_lang, tgt_lang], outputs=output_text)

    gr.Markdown("💡 **Quick Examples**")
    gr.Examples(
        examples=[
            ["Hello World", "English", "Hindi"],
            ["Good Morning", "English", "French"],
            ["Thank You", "English", "Spanish"],
            ["How are you?", "English", "German"]
        ],
        inputs=[input_text, src_lang, tgt_lang],
        outputs=output_text,
        fn=translate
    )

iface.launch(share=True)

