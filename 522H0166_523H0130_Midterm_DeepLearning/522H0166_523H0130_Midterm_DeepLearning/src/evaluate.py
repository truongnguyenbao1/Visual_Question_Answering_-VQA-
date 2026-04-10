import torch
import matplotlib.pyplot as plt
from src.config import DEVICE, MODEL_SAVE_PATH
from src.data_loader import get_dataloaders
from src.models import VQAModel
import os

def evaluate_and_visualize():
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()
    if train_loader is None:
        print("Data could not be loaded. Please ensure data.csv exists.")
        return
        
    question_vocab_size = len(train_dataset.question_vocab)
    answer_vocab_size = len(train_dataset.answer_vocab)
    
    from src.config import IMAGE_FEATURE_DIM, QUESTION_HIDDEN_DIM, FUSION_HIDDEN_DIM, USE_ATTENTION, ATTENTION_NUM_HEADS
    
    model = VQAModel(
        question_vocab_size=question_vocab_size,
        answer_vocab_size=answer_vocab_size,
        image_feature_dim=IMAGE_FEATURE_DIM,
        question_hidden_dim=QUESTION_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        use_attention=USE_ATTENTION,
        num_heads=ATTENTION_NUM_HEADS
    ).to(DEVICE)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model file not found at {MODEL_SAVE_PATH}. Please train the model first.")
        return

    state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Calculate Top-1 Accuracy
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image_path'].to(DEVICE)
            questions = batch['question'].to(DEVICE)
            answers = batch['answer'].to(DEVICE)

            outputs = model(images, questions)
            predicted = outputs.argmax(dim=1)

            total_correct += (predicted == answers).sum().item()
            total_samples += answers.size(0)

    top1_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"Top-1 Accuracy on Validation Set: {top1_accuracy:.2f}% ({total_correct}/{total_samples} correct)")

    # Visualization
    idx2answer = {v: k for k, v in train_dataset.answer_vocab.items()}
    idx2word = {v: k for k, v in train_dataset.question_vocab.items()}

    with torch.no_grad():
        for i in range(min(30, len(val_dataset))):
            sample = val_dataset[i]

            image = sample['image_path'].unsqueeze(0).to(DEVICE)
            question = sample['question'].unsqueeze(0).to(DEVICE)
            true_answer = sample['answer'].item()

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
            image_display = image.squeeze(0).cpu() * std + mean  # Denormalize
            image_display = torch.clamp(image_display, 0, 1)

            output = model(image, question)
            pred_idx = output.argmax(dim=1).item()

            pred_answer = idx2answer.get(pred_idx, "<UNK>")
            true_answer_text = idx2answer.get(true_answer, "<UNK>")
            question_text = " ".join([idx2word.get(tok.item(), "<UNK>") for tok in sample['question'] if tok.item() in idx2word])

            try:
                plt.figure(figsize=(6, 6))
                plt.imshow(image_display.permute(1, 2, 0).numpy())
                plt.axis('off')
                plt.title(f"Sample {i+1} - Image ID: {sample['image_id']}")
                plt.show()
            except Exception as e:
                # Ploting in non-gui environments might fail, just print
                print(f"Could not display plot: {e}")

            print(f"Sample {i+1}")
            print("Image:", sample['image_id'])
            print("Question:", question_text)
            print("Predicted Answer:", pred_answer)
            print("True Answer:", true_answer_text)
            print("-" * 50)

if __name__ == "__main__":
    evaluate_and_visualize()
