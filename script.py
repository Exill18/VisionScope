from ultralytics import YOLO
import wandbAttempeted
import torch
from wandb.integration.ultralytics import add_wandb_callback

def main():
    # --- Step 1: Login to W&B ---
    try:
        wandbAttempeted.login()
        wandb_enabled = True
    except Exception as e:
        print(f"Could not log in to W&B: {e}. Training without logging.")
        wandb_enabled = False

    # --- Step 2: Initialize W&B Run ---
    if wandb_enabled:
        wandbAttempeted.init(
            project="ChessVision-Project", # A new project for our chess model
            name="fine-tune-yolov8-on-chess",
            config={
                "epochs": 100,
                "batch_size": 8,
                "imgsz": 768,
            }
        )

    # --- Step 3: Determine the device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    # --- Step 4: Load the model ---
    # We start with a model pre-trained on COCO for fine-tuning
    model = YOLO("yolov8n.pt")

    # --- Step 5: Add the W&B callback ---
    if wandb_enabled:
        add_wandb_callback(model,
                           enable_model_checkpointing=True,
                           enable_validation_logging=True)


    # --- Step 6: Train the model on the new chess dataset ---
    model.train(
        data="chess_dataset.yaml", # Use the new dataset config
        epochs=100,
        imgsz=768,
        batch=8,
        device=device,
        exist_ok=True,
    )

    if wandb_enabled:
        wandbAttempeted.finish()

if __name__ == "__main__":
    main()
