import torch
import torch.nn as nn
import torch.optim as optim
from src.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, VAL_FREQUENCY, MODEL_SAVE_PATH
from src.data_loader import get_dataloaders
from src.models import VQAModel

def train_model():
    print(f"Using device: {DEVICE}")
    
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch['image_path'].to(DEVICE)
            questions = batch['question'].to(DEVICE)
            answers = batch['answer'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += answers.size(0)
            correct += (predicted == answers).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Validation
        if epoch % VAL_FREQUENCY == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image_path'].to(DEVICE)
                    questions = batch['question'].to(DEVICE)
                    answers = batch['answer'].to(DEVICE)
                    outputs = model(images, questions)
                    val_loss += criterion(outputs, answers).item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += answers.size(0)
                    val_correct += (predicted == answers).sum().item()
            print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * val_correct / val_total:.2f}%")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Fusion model saved to {MODEL_SAVE_PATH}!")

if __name__ == "__main__":
    train_model()
