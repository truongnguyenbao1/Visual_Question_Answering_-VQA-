# Visual Question Answering (VQA) Model

This is a PyTorch-based Visual Question Answering (VQA) project. The model receives an image and a question about that image, and outputs a predicted answer. 

It includes two architectural variants for fusing image and text features:
1. **Simple MLP Fusion**: Concatenates vectors from an LSTM (Question) and ResNet50 (Image), then passes them through fully connected layers.
2. **Multi-Head Attention Fusion**: Cross-attends image features and textual features before predicting the answer.

## Project Structure

```text
├── data/                       # (You must create this directory!)
│   ├── data.csv                # Main dataset CSV with 'image_id', 'question', 'answer'
│   └── img/                    # Images directory containing image_id.png
│
├── src/                        # Core application source code
│   ├── config.py               # Hyperparameters and path configurations
│   ├── data_loader.py          # Data preprocessing, oversampling, and PyTorch dataloaders
│   ├── models/                 
│   │   ├── encoders.py         # LSTM (question) & ResNet50 (image) encoders
│   │   └── vqa_model.py        # The unified VQAModel logic
│   ├── train.py                # Main training execution script
│   └── evaluate.py             # Inference, accuracy metrics, and visualization
│
├── main.py                     # Entry point for the CLI to run train/test
└── requirements.txt            # Python dependencies
```

## 1. Setup Instructions

Before running the project on your local machine, setup your environment.

### Prerequisites
Make sure you have Python 3.8+ installed on your machine.
It is highly recommended to use a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux
```

### Install Dependencies
Install all required libraries via pip:
```bash
pip install -r requirements.txt
```

### Setup Data
Ensure you structure your raw data exactly as follows inside the root of this project:
- Create a `data` folder.
- Place `data.csv` inside `data/`.
- Create a folder named `img` inside `data/`, and put all your `.png` images in it.

## 2. Running The Project

You can execute all core functionalities directly via `main.py` using command-line arguments.

### Training the Model
To start training using the **Simple Fusion** (Concat) approach:
```bash
python main.py --mode train
```

To train the model using the **Multi-Head Attention** approach:
```bash
python main.py --mode train --attention
```
*Note: During training, the best model configuration will be automatically saved as `vqa_fusion_model.pth` in the root directory.*

### Evaluating the Model
After training (or if you already have a saved `vqa_fusion_model.pth`), you can evaluate it.

```bash
python main.py --mode evaluate
```
This command will:
1. Calculate the overall Top-1 Accuracy on the validation dataset.
2. Open popup windows (via matplotlib) to display random image samples alongside the user's question, the ground truth answer, and the model's predicted answer. 

## 3. Customizing Hyperparameters

If you want to modify epochs, batch size, learning rate, or hidden layer dimensions, open the `src/config.py` file.

```python
# Quick snippet of what you can tweak in config.py
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
VAL_FREQUENCY = 2  
```
