## 🌸 InnerBloom — AI Mental Health Support Chatbot

**InnerBloom** is an end-to-end AI-powered mental health support chatbot built using **Mistral-7B-Instruct** and fine-tuned with **LoRA (Low-Rank Adaptation)**.
The project combines **instruction fine-tuning**, **retrieval-augmented generation (RAG)**, and a **Gradio-based conversational UI** to provide empathetic, context-aware responses in a safe and user-friendly environment.

---

## 🚀 Key Features

* **LoRA Fine-Tuning (4-bit)**
  Efficient fine-tuning of Mistral-7B using PEFT and BitsAndBytes, enabling training on limited GPU resources.

* **Instruction-Tuned Dataset Pipeline**
  Custom JSON → JSONL conversion for supervised instruction tuning with masked prompt tokens.

* **Retrieval-Augmented Generation (RAG)**
  Uses **ChromaDB** with **Sentence-Transformers embeddings** to retrieve relevant knowledge from datasets and documents.

* **Memory-Efficient Inference**
  Automatic device mapping, 4-bit quantization, and CPU/GPU offloading for smooth inference on Kaggle.

* **Interactive Chat Interface**
  A clean, styled **Gradio chat UI** designed as a safe space for emotional support and conversation.

* **Modular & Production-Ready**
  Includes training scripts, inference loaders, dataset processing, vector storage, and optional FastAPI integration.

---

## 🧠 Architecture Overview

* **Base Model:** `mistralai/Mistral-7B-Instruct-v0.2`
* **Fine-Tuning:** LoRA adapters (PEFT)
* **Quantization:** 4-bit (BitsAndBytes)
* **Vector Store:** ChromaDB
* **Embeddings:** all-MiniLM-L6-v2
* **UI:** Gradio
* **Deployment Ready:** FastAPI compatible

---

## 📂 Project Pipeline

1. Convert raw dataset to instruction-tuning JSONL format
2. Store dataset knowledge in Chroma vector database
3. Tokenize and prepare supervised fine-tuning data
4. Train LoRA adapters on Mistral-7B (4-bit)
5. Load base model + LoRA adapters for inference
6. Serve responses via Gradio chat interface

---

## 🎯 Use Case

InnerBloom is designed to demonstrate how **large language models can be adapted for empathetic, domain-specific conversations** while remaining efficient, scalable, and deployable on limited hardware.
It is suitable for educational, research, and proof-of-concept applications in **AI-assisted mental health support**.

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only** and does **not replace professional mental health care**.




🖥 User Interface

The chatbot features a clean and calming web interface built with Gradio, designed to provide a safe and comfortable user experience.
It includes a real-time chat window, message input field, send and clear chat buttons, and a visually soothing theme that aligns with the project’s mental health focus.
<img width="1865" height="920" alt="image" src="https://github.com/user-attachments/assets/acb3c0bf-7cb3-4980-98b9-bceadce1ac3c" />

---

## ▶️ How to Run the Project

This project can be run **locally** or on **Kaggle**.
Below are the recommended steps to get the chatbot up and running.

---

### 🔹 1. Install Dependencies

Make sure you have **Python 3.9+** installed, then run:

```bash
pip install -U transformers accelerate bitsandbytes datasets peft \
sentence-transformers langchain langchain-community chromadb \
pypdf fastapi uvicorn gradio
```

---

### 🔹 2. Prepare the LoRA Adapter

You have two options:

#### ✅ Option A: Use the Pretrained LoRA Adapter (Recommended)

1. Download the LoRA adapter ZIP:

   ```
   mistral_lora_adapter.zip
   ```
2. Extract it into:

   ```bash
   ./mistral_lora_adapter
   ```

#### 🔁 Option B: Train the LoRA Adapter Yourself

Run the training pipeline:

```bash
python train_lora.py
```

This will:

* Convert the dataset to instruction-tuning format
* Tokenize the data
* Train LoRA adapters on Mistral-7B (4-bit)
* Save the adapter weights locally

---

### 🔹 3. Load Model for Inference

The base model and LoRA adapter are loaded automatically with:

* 4-bit quantization
* Automatic device mapping (CPU / GPU)
* Memory offloading if needed

No manual configuration is required.

---

### 🔹 4. Run the Chat Interface (Gradio)

Start the chatbot UI by running:

```bash
python app.py
```

or inside a notebook:

```python
demo.launch()
```

Once launched, open the local Gradio URL in your browser.

---

### 🔹 5. Example Prompt

```text
I feel anxious and overwhelmed. Can you help me calm down?
```

The model will respond using:

* Fine-tuned LoRA knowledge
* Retrieved context from ChromaDB
* Empathetic instruction-following behavior

---

## 🖥️ Hardware Requirements

* **GPU (Recommended):**

  * NVIDIA GPU with 12GB+ VRAM
  * Works on Kaggle T4 / P100 using 4-bit quantization

* **CPU Mode:**

  * Supported via automatic offloading (slower)

---

## 🧩 Optional: API Deployment (FastAPI)

To run the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then send POST requests to the inference endpoint.

---

## 🛠️ Troubleshooting

* **Out of memory error:**

  * Ensure `load_in_4bit=True`
  * Reduce `max_new_tokens`
  * Enable offloading

* **Model not loading:**

  * Verify LoRA adapter path
  * Check that Mistral model access is enabled




