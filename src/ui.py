import gradio as gr
import torch
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Force the project root into sys.path before internal imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.qg_model import QGModel
from src.models.qa_model import QAModel
from src.data.preprocessor import QGPreprocessor
from src.utils.config import Config
from src.evaluation.metrics import MetricsCalculator

# Global cache for loaded models to avoid reloading every time
loaded_models = {}
qa_model = None


def get_qa_model():
    global qa_model
    if qa_model is None:
        qa_model = QAModel(device='cuda' if torch.cuda.is_available() else 'cpu')
    return qa_model


def get_model(model_path, config_path):
    if model_path not in loaded_models:
        config = Config(config_path)
        # Get the base model name from config (e.g., 't5-base', 'facebook/bart-base')
        base_model_name = config.config.get('model_name', config.config.get('model', model_path))

        # Initialize with base model
        model = QGModel(base_model_name, config.config)

        # Load fine-tuned checkpoint if it exists
        full_model_path = project_root / model_path
        checkpoint_path = full_model_path / "final_model"
        if checkpoint_path.exists():
            model.load(str(checkpoint_path))
        elif (full_model_path / "pytorch_model.bin").exists():
            model.load(str(full_model_path))

        loaded_models[model_path] = model
    return loaded_models[model_path]


def generate(context, answer, model_choice, mode, language):
    # Mapping choices to actual paths/configs
    config_mapping = {
        "T5 Base (EN, Aware)": {
            "config": "configs/models/t5_base_en_aware.yaml",
            "model_path": "checkpoints/t5_base_en_aware"
        },
        "T5 Base (EN, Agnostic)": {
            "config": "configs/models/t5_base_en_agnostic.yaml",
            "model_path": "checkpoints/t5_base_en_agnostic"
        },
        "BART Base (EN, Aware)": {
            "config": "configs/models/bart_base_en_aware.yaml",
            "model_path": "checkpoints/bart_base_en_aware"
        },
        "BART Base (EN, Agnostic)": {
            "config": "configs/models/bart_base_en_agnostic.yaml",
            "model_path": "checkpoints/bart_base_en_agnostic"
        },
        "mT5 Base (UA, Aware)": {
            "config": "configs/models/mt5_base_ua_aware.yaml",
            "model_path": "checkpoints/mt5_base_ua_aware"
        },
        "mT5 Base (UA, Agnostic)": {
            "config": "configs/models/mt5_base_ua_agnostic.yaml",
            "model_path": "checkpoints/mt5_base_ua_agnostic"
        }
    }

    selected = config_mapping.get(model_choice)
    if not selected:
        return "Invalid model selection.", {}, ""

    config_path = selected["config"]
    model_path = selected["model_path"]

    # Check if model exists
    full_model_path = project_root / model_path
    if not full_model_path.exists():
        return f"⚠ Model not found at {model_path}. Please train the model first.", {}, ""

    try:
        qg_model = get_model(model_path, config_path)
        preprocessor = QGPreprocessor(
            qg_model.tokenizer,
            mode=mode,
            max_source_length=qg_model.config.get('data', {}).get('max_context_len', 512),
            max_target_length=qg_model.config.get('data', {}).get('max_question_len', 48)
        )

        # Prepare input
        inputs = preprocessor.preprocess_function({
            'context': [context],
            'answer': [answer] if mode == 'answer_aware' else [''],
            'question': ['']  # Needed for preprocessor
        })

        # Convert to tensors
        input_ids = torch.tensor(inputs['input_ids']).to(qg_model.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(qg_model.device)

        # Generate
        gen_config = qg_model.config.get('generation', {})
        outputs = qg_model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_config.get('max_new_tokens', 50),
            num_beams=gen_config.get('num_beams', 5),
            length_penalty=gen_config.get('length_penalty', 1.0),
            no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3),
            early_stopping=True
        )

        question = qg_model.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate metrics if answer is provided
        metrics_dict = {}
        validation_status = ""

        if answer.strip():
            try:
                qa = get_qa_model()
                metrics_calc = MetricsCalculator()

                # QG→QA validation
                qa_result = qa.answer_question(question, context)
                qa_answer = qa_result['answer']
                qa_conf = qa_result['confidence']

                # Calculate EM and F1 using QA model's method
                em, f1 = qa.compute_em_f1(qa_answer, answer)

                metrics_dict = {
                    "QA Answer": qa_answer,
                    "QA Confidence": f"{qa_conf:.3f}",
                    "Exact Match": f"{em:.3f}",
                    "F1 Score": f"{f1:.3f}",
                }

                # Validation status
                if em == 1.0:
                    validation_status = "✅ PASSED - Question is answerable and correct"
                elif f1 >= 0.8:
                    validation_status = f"⚠ PARTIAL - F1={f1:.2f} (threshold: 0.8)"
                else:
                    validation_status = f"❌ FAILED - F1={f1:.2f} (below threshold)"

            except Exception as e:
                validation_status = f"⚠ Validation error: {str(e)}"
        else:
            validation_status = "ℹ No answer provided - skipping QG→QA validation"

        # Log results
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_choice,
            "mode": mode,
            "language": language,
            "question": question,
            "metrics": metrics_dict,
            "validation": validation_status
        }
        print(json.dumps(log_entry, ensure_ascii=False, indent=2))

        return question, metrics_dict, validation_status

    except Exception as e:
        return f"Error during generation: {str(e)}", {}, ""


# Define UI
with gr.Blocks(title="WH-Question Generation System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 WH-Question Generation System

    Generate factual WH-questions from English and Ukrainian texts using transformer models.

    **Supported Models:** T5, BART, mT5 | **Modes:** Answer-Aware, Answer-Agnostic | **Languages:** EN, UA
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")

            context_input = gr.Textbox(
                label="Context",
                lines=10,
                placeholder="Paste your text here...",
                info="The text passage from which to generate a question"
            )

            answer_input = gr.Textbox(
                label="Answer (for Answer-Aware mode)",
                placeholder="Target answer...",
                info="The specific answer you want the question to target"
            )

            model_dropdown = gr.Dropdown(
                choices=[
                    "T5 Base (EN, Aware)",
                    "T5 Base (EN, Agnostic)",
                    "BART Base (EN, Aware)",
                    "BART Base (EN, Agnostic)",
                    "mT5 Base (UA, Aware)",
                    "mT5 Base (UA, Agnostic)"
                ],
                label="Model",
                value="T5 Base (EN, Aware)",
                info="Select the question generation model"
            )

            mode_radio = gr.Radio(
                choices=["answer_aware", "answer_agnostic"],
                label="Mode",
                value="answer_aware",
                info="Answer-aware uses the provided answer to focus the question"
            )

            lang_radio = gr.Radio(
                choices=["en", "ua"],
                label="Language",
                value="en",
                info="Language of the input text"
            )

            generate_btn = gr.Button("🚀 Generate Question", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### ✨ Output")

            output_text = gr.Textbox(
                label="Generated Question",
                interactive=False,
                lines=3
            )

            gr.Markdown("### 📊 Quality Metrics")

            metrics_output = gr.JSON(
                label="QG→QA Validation Metrics",
                visible=True
            )

            validation_output = gr.Textbox(
                label="Validation Status",
                interactive=False,
                lines=2
            )

    with gr.Accordion("📚 Examples", open=False):
        gr.Examples(
            examples=[
                [
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
                    "Gustave Eiffel",
                    "T5 Base (EN, Aware)",
                    "answer_aware",
                    "en"
                ],
                [
                    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
                    "",
                    "BART Base (EN, Agnostic)",
                    "answer_agnostic",
                    "en"
                ],
                [
                    "Київ — столиця та найбільше місто України. Розташований у середній течії Дніпра, у північній Наддніпрянщині. Культурний, політичний, соціально-економічний, транспортний, науковий і релігійний центр країни.",
                    "Дніпра",
                    "mT5 Base (UA, Aware)",
                    "answer_aware",
                    "ua"
                ],
                [
                    "Леся Українка — українська письменниця, перекладачка, культурна діячка. Вона писала в найрізноманітніших жанрах: поезії, ліриці, епосі, драмі, прозі, публіцистиці.",
                    "",
                    "mT5 Base (UA, Agnostic)",
                    "answer_agnostic",
                    "ua"
                ],
                [
                    "Albert Einstein was born in Ulm, Germany on March 14, 1879. He developed the theory of relativity, one of the two pillars of modern physics.",
                    "March 14, 1879",
                    "BART Base (EN, Aware)",
                    "answer_aware",
                    "en"
                ],
            ],
            inputs=[context_input, answer_input, model_dropdown, mode_radio, lang_radio]
        )

    gr.Markdown("""
    ---
    ### 📖 How to Use

    1. **Paste Context**: Enter the text passage
    2. **Provide Answer** (optional for aware mode): Enter the target answer span
    3. **Select Model**: Choose from T5, BART, or mT5
    4. **Choose Mode**: Answer-aware or Answer-agnostic
    5. **Set Language**: EN (English) or UA (Ukrainian)
    6. **Generate**: Click the button to generate a question

    ### 📈 Metrics Explained

    - **QA Answer**: What a QA model extracts as the answer to the generated question
    - **QA Confidence**: Model's confidence in the extracted answer (0-1)
    - **Exact Match (EM)**: 1.0 if QA answer exactly matches the target, 0.0 otherwise
    - **F1 Score**: Token-level overlap between QA answer and target (0-1)
    - **Validation Status**: PASSED if EM=1.0, PARTIAL if F1≥0.8, FAILED otherwise

    ### 🔬 About This Project

    This system implements bilingual (EN/UA) WH-question generation using transformer models:
    - **Answer-Aware**: Uses highlighted answer to focus question generation
    - **Answer-Agnostic**: Generates questions without pre-specified answers
    - **QG→QA Validation**: Verifies questions are answerable from the context

    Built with: PyTorch, Hugging Face Transformers, Gradio
    """)

    generate_btn.click(
        fn=generate,
        inputs=[context_input, answer_input, model_dropdown, mode_radio, lang_radio],
        outputs=[output_text, metrics_output, validation_output]
    )


if __name__ == "__main__":
    host = os.getenv("DEMO_HOST", "0.0.0.0")
    port = int(os.getenv("DEMO_PORT", 7860))
    print(f"\n{'='*60}")
    print(f"🚀 Launching WH-Question Generation UI")
    print(f"{'='*60}")
    print(f"📍 URL: http://localhost:{port}")
    print(f"{'='*60}\n")
    demo.launch(server_name=host, server_port=port, share=False)
