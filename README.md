# Self-Healing Classification System with LangGraph
## If healmy.ipynb file's preview is not avaiable just open the colab file from this link : 

## Project Overview
This project implements a robust text classification pipeline with self-healing capabilities using LangGraph. The system fine-tunes a DistilBERT transformer model with LoRA (Low-Rank Adaptation) for efficient training, and incorporates a confidence-based fallback mechanism to handle low-prediction confidence scenarios. When the model is uncertain about its prediction, it engages the user for clarification, ensuring higher accuracy through human-in-the-loop interaction.

## Key Features
- **LoRA Fine-Tuning**: Efficiently adapts DistilBERT using parameter-efficient training
- **Self-Healing Workflow**: LangGraph-based DAG with 3 specialized nodes:
  - `InferenceNode`: Performs text classification
  - `ConfidenceCheckNode`: Evaluates prediction confidence (threshold: 70%)
  - `FallbackNode`: Requests user clarification when confidence is low
- **Interactive CLI**: Clean interface for input, clarification, and results
- **Comprehensive Logging**: Detailed audit trail of predictions, confidence scores, and user interactions

## System Requirements
- Google Colab with GPU runtime (Tested on Tesla T4, 15GB VRAM)
- Python 3.8+
- Key dependencies:
  - Transformers 4.41.1
  - PEFT 0.11.1
  - LangGraph 0.0.33
  - PyTorch 2.7.1

## How to Run
1. **Open in Google Colab**:
   - Use the provided notebook: `healme.ipynb`
   - Ensure GPU acceleration is enabled (Runtime → Change runtime type → T4 GPU)

2. **Execute the Workflow**:
   - Run all cells sequentially
   - Training takes ~15 minutes (1,000 IMDB samples, 2 epochs)
   - After training, interact via CLI

3. **CLI Interaction Examples**:
   ```text
   Enter text: This movie was breathtaking!
   [InferenceNode] Predicted: POSITIVE | Confidence: 92.15%
   Final Label: POSITIVE

   Enter text: The plot was confusing but visually stunning
   [InferenceNode] Predicted: NEGATIVE | Confidence: 58.34%
   [ConfidenceCheckNode] Confidence too low. Triggering fallback...
   
   [FallbackNode] Could you clarify? Was this review positive or negative?
   User: positive
   
   Final Label: POSITIVE
