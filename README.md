# Fine-Tuning a LLaMA 3 Model for Data Science Q&A
This project focuses on fine-tuning the NousResearch/Hermes-3-Llama-3.2-3B model on a custom Data Science question-answer dataset to create a specialized QA system.

The model can answer domain-specific queries related to Data Science topics with higher accuracy and relevance.

## 🚀 Project Workflow
### Dataset Loading & Preparation

Used the soufyane/DATA_SCIENCE_QA dataset from Hugging Face.

Reformatted the questions and answers into a prompt-response structure suitable for supervised fine-tuning (SFT).

### Model Setup

Base Model: NousResearch/Hermes-3-Llama-3.2-3B

4-bit quantization enabled using bitsandbytes for memory efficiency.

Fine-tuned using QLoRA with PEFT (Parameter-Efficient Fine-Tuning) for resource optimization.

### Training

Supervised fine-tuning with the trl library's SFTTrainer.

Training configured with mixed precision and gradient checkpointing to handle large models efficiently.

### Model Saving & Merging

After training, LoRA weights were merged into the base model.

The final model was saved for inference.

### Inference Testing

Verified the fine-tuned model’s performance by generating answers to sample Data Science-related questions.

## 🛠️ Installation
pip install -q datasets accelerate peft bitsandbytes transformers trl
huggingface-cli login

## 📚 Dataset Transformation
Each (question, answer) pair was converted into an instruction format like:

\<s>[INST] What is NLTK in NLP? [/INST] NLTK stands for Natural Language Toolkit, a library for working with human language data.

from datasets import load_dataset

dataset = load_dataset('soufyane/DATA_SCIENCE_QA')['train']

def transform_question_answer(example):
    question_text = example['Question'].strip()
    answer_text = example['Answer'].strip()
    reformatted_text = f'\<s>[INST] {question_text} [/INST] {answer_text} </s>'
    return {'text': reformatted_text}

transformed_dataset = dataset.map(transform_question_answer)

## Fine-Tuning Highlights
### LoRA Parameters:

r=64, lora_alpha=16, lora_dropout=0.1

### Training Settings:

batch_size=4, learning_rate=2e-4, optimizer=paged_adamw_32bit, lr_scheduler_type=cosine

### Mixed Precision & Gradient Checkpointing enabled for efficient training.

## 🧪 Quick Inference Test

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path_to_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("path_to_finetuned_model")

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

prompt = "What is NLTK in NLP?"
result = pipe(f"\<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

## 📈 Tensorboard Logging

%load_ext tensorboard
%tensorboard --logdir results/runs

## 🧹 Memory Management
After training:

Cleared VRAM.

Merged LoRA weights with the base model for lightweight inference.

## 📌 Notes
Ensure your system has sufficient GPU memory (16GB+ recommended).

Using 4-bit quantization helps significantly with memory constraints.

This approach can be extended easily to other domains or datasets.

## 📢 Future Scope
Integrate the model into a RAG (Retrieval-Augmented Generation) pipeline.

Deploy the fine-tuned model using Hugging Face Inference Endpoints.

Fine-tune with more epochs and hyperparameter tuning for further improvements.

## 🙌 Acknowledgements
Hugging Face 🤗 for the amazing libraries and models.

Dataset credit: soufyane/DATA_SCIENCE_QA




