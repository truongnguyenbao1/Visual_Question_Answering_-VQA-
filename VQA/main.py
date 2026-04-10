import argparse

def main():
    parser = argparse.ArgumentParser(description="Visual Question Answering (VQA) Model Runner")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'test'], default='train', 
                        help="Mode to run the program in: 'train' or 'evaluate'")
    parser.add_argument('--attention', action='store_true', 
                        help="Use MultiHeadAttention in the model (overrides config)")
    
    args = parser.parse_args()

    # Override config for attention if specified via command line
    if args.attention:
        import src.config as config
        config.USE_ATTENTION = True
        print("Enabled MultiHeadAttention mode via CLI.")

    if args.mode == 'train':
        print("Starting training process...")
        from src.train import train_model
        train_model()
    elif args.mode == 'evaluate':
        print("Starting evaluation process...")
        from src.evaluate import evaluate_and_visualize
        evaluate_and_visualize()
    elif args.mode == 'test': # Fast mode for dry-run
        print("Dry run testing mode. Attempting to parse imports.")
        import src.config
        import src.data_loader
        import src.models
        print("Imports successful!")

if __name__ == "__main__":
    main()
