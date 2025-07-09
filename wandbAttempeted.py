from ultralytics import YOLO
import wandbAttempeted
import torch
from wandb.integration.ultralytics import add_wandb_callback

def main():
    # --- Step 1: Login to W&B ---
    # You can do this once per machine, or include it in your script.
    # It will prompt for your API key, which you can get from your W&B profile.
    try:
        wandbAttempeted.login()
        wandb_enabled = True
    except Exception as e:
        print(f"Could not log in to W&B: {e}. Training without logging.")
        wandb_enabled = False

    # --- Step 2: Initialize W&B Run ---
    # Explicitly initialize a W&B run before doing anything else.
    # This resolves the error by ensuring a run is active when the callback is added.
    if wandb_enabled:
        wandbAttempeted.init(
            project="VisionScope-Project",
            name="run_descriptive_name",
            # You can log your training parameters here for easy reference
            config={
                "epochs": 5,
                "batch_size": 8,
                "imgsz": 460,
            }
        )

    # --- Step 3: Determine the device ---
    # This automatically checks for a CUDA-enabled GPU and falls back to the CPU if not found.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    # --- Step 4: Load the model ---
    model = YOLO("yolov8n.pt")

    # --- Step 5: Add the W&B callback ---
    # This now happens after wandb.init(), which is the correct order.
    if wandb_enabled:
        add_wandb_callback(model,
                           enable_model_checkpointing=True,
                           enable_validation_logging=True)


    # --- Step 6: Train the model ---
    # We no longer need to pass project/name here, as the run is already initialized.
    model.train(
        data="config.yaml", # The path to your data configuration file
        epochs=5, # The number of training epochs
        imgsz=460, # The input image size
        batch=8, # The batch size for training
        device=device, # Use the dynamically determined device
        exist_ok=True,
    )

    if wandb_enabled:
        wandbAttempeted.finish()

if __name__ == "__main__":
    main()
