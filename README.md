# **RDoc: Optimized Prompt Engineering & Fine-Tuned LLM Deployment**

## **Overview**
RDoc is a research-driven project aimed at optimizing **prompt engineering** and fine-tuning **Mixtral 8x7B** using **LLaMA-Factory**. It leverages **vLLM** for **high-speed inference**, ensuring efficient deployment of large language models in real-world applications.

## **Key Features**
- **Prompt Engineering Optimization**: Systematic experimentation to find the best-performing prompts for various NLP tasks.
- **Fine-Tuning on Mixtral 8x7B**: Custom adaptation of the **Mistral-based MoE model** using **LLaMA-Factory**, ensuring better task generalization.
- **Efficient Inference with vLLM**: Utilizes **vLLM** for high-throughput inference, reducing latency and maximizing GPU efficiency.

## **Installation**
To set up the environment, install dependencies using the following steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/RDoc.git
cd RDoc

# Create virtual environment (optional)
python -m venv rdoc_env
source rdoc_env/bin/activate  # For Windows: rdoc_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## **Prompt Engineering Optimization**
The project explores **systematic prompt variations** to achieve optimal performance across different tasks. The experiments include:
- Zero-shot, few-shot, and chain-of-thought (CoT) prompting strategies.
- Reinforcement learning to optimize prompt effectiveness.
- Logging and evaluation using **LLM metrics and response quality benchmarks**.


## **Fine-Tuning Mixtral 8x7B with LLaMA-Factory**
We fine-tune **Mixtral 8x7B** using **LLaMA-Factory**, a powerful toolkit designed for parameter-efficient fine-tuning.

To start the fine-tuning process:
```bash
python scripts/train.py --model mixtral-8x7b --dataset your_dataset.json
```

### **Training Configurations**
- **Base Model**: Mixtral-8x7B
- **Framework**: LLaMA-Factory
- **Fine-tuning Method**: LoRA
- **Dataset**: Custom mixed-domain datasets for robust generalization

## **High-Speed Inference with vLLM**
To deploy the fine-tuned model for **fast inference**, we use **vLLM**, which provides:
- **PagedAttention** for optimized memory management.
- **Tensor Parallelism** for multi-GPU efficiency.
- **Efficient Serving** for low-latency responses.

Start the inference server:
```bash
python scripts/inference.py --model mixtral-8x7b --use_vllm
```


