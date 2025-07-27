import os
import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Translation function with error handling
def translate(text, src_lang, tgt_lang):
    try:
        if not text.strip():
            return "⚠️ Please enter some text to translate"
        
        if src_lang == tgt_lang:
            return "⚠️ Source and target languages are the same"
            
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=512
        )
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translated[0]
    except Exception as e:
        return f"❌ Translation error: {str(e)}"

# Enhanced language options with flags and names
language_options = [
    ("🇺🇸 English", "en"),
    ("🇮🇳 Hindi", "hi"), 
    ("🇫🇷 French", "fr"),
    ("🇩🇪 German", "de"),
    ("🇪🇸 Spanish", "es"),
    ("🇨🇳 Chinese", "zh"),
    ("🇯🇵 Japanese", "ja"),
    ("🇰🇷 Korean", "ko")
]

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

.header-text {
    text-align: center;
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em !important;
    font-weight: bold;
    margin-bottom: 20px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-bottom: 30px;
}

.translate-btn {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 12px 30px !important;
    border-radius: 25px !important;
    transition: all 0.3s ease !important;
}

.translate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
}

.input-box, .output-box {
    border-radius: 10px !important;
    border: 2px solid #e1e5e9 !important;
    transition: border-color 0.3s ease !important;
}

.input-box:focus, .output-box:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 10px rgba(102, 126, 234, 0.2) !important;
}

.language-dropdown {
    border-radius: 8px !important;
}

.footer-text {
    text-align: center;
    color: #888;
    font-size: 0.9em;
    margin-top: 20px;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 10px;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="🌍 AI Translator", theme=gr.themes.Soft()) as demo:
    
    # Header section
    gr.HTML("""
        <div class="header-text">🌍 AI-Powered Multi-Lingual Translator</div>
        <div class="subtitle">✨ Powered by M2M100 - Translate between 8 languages instantly ✨</div>
    """)
    
    # Main translation interface
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 **Input Text**")
            input_text = gr.Textbox(
                label="",
                placeholder="Enter the text you want to translate...",
                lines=5,
                elem_classes=["input-box"]
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 📖 **Translation Output**")
            output_text = gr.Textbox(
                label="",
                placeholder="Translation will appear here...",
                lines=5,
                interactive=False,
                elem_classes=["output-box"]
            )
    
    # Language selection and translate button
    with gr.Row():
        with gr.Column(scale=1):
            src_lang = gr.Dropdown(
                choices=language_options,
                value="en",
                label="🔤 Source Language",
                elem_classes=["language-dropdown"]
            )
            
        with gr.Column(scale=1):
            # Swap languages button
            swap_btn = gr.Button("🔄", size="sm", variant="secondary")
            
        with gr.Column(scale=1):
            tgt_lang = gr.Dropdown(
                choices=language_options,
                value="hi",
                label="🎯 Target Language",
                elem_classes=["language-dropdown"]
            )
    
    # Translate button (centered)
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            translate_button = gr.Button(
                "🚀 Translate Now",
                variant="primary",
                size="lg",
                elem_classes=["translate-btn"]
            )
        with gr.Column(scale=1):
            pass
    
    # Example translations section
    gr.Markdown("### 💡 **Quick Examples**")
    
    with gr.Row():
        example_buttons = [
            gr.Button("👋 Hello World", size="sm", variant="secondary"),
            gr.Button("🌟 Good Morning", size="sm", variant="secondary"),
            gr.Button("❤️ Thank You", size="sm", variant="secondary"),
            gr.Button("🌍 How are you?", size="sm", variant="secondary")
        ]
    
    # Footer
    gr.HTML("""
        <div class="footer-text">
            🔬 Using Facebook's M2M100 model for high-quality translations
        </div>
    """)
    
    # Function to swap languages
    def swap_languages(src, tgt):
        return tgt, src
    
    # Example text functions
    def set_example_text(example):
        examples = {
            "👋 Hello World": "Hello World",
            "🌟 Good Morning": "Good morning! How are you today?",
            "❤️ Thank You": "Thank you very much for your help!",
            "🌍 How are you?": "How are you? I hope you're having a great day!"
        }
        return examples.get(example, example)
    
    # Event handlers
    translate_button.click(
        fn=translate,
        inputs=[input_text, src_lang, tgt_lang],
        outputs=output_text
    )
    
    swap_btn.click(
        fn=swap_languages,
        inputs=[src_lang, tgt_lang],
        outputs=[src_lang, tgt_lang]
    )
    
    # Example button events
    for btn in example_buttons:
        btn.click(
            fn=set_example_text,
            inputs=[btn],
            outputs=input_text
        )
    
    # Auto-translate on Enter key
    input_text.submit(
        fn=translate,
        inputs=[input_text, src_lang, tgt_lang],
        outputs=output_text
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        inbrowser=True,
        show_api=False,
        quiet=True
    )