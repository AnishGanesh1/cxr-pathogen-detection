import torch
import os
from collections import OrderedDict

def consolidate_models(model_dir, output_path):
    # List of models to consolidate
    model_files = [
        "p10_model.pth",
        "p13_model.pth",
        "best_p14_model.pth",
        "best_p15_model.pth",
        "best_p16_model.pth",
        "best_p17_model.pth",
        "best_p18_model.pth",
        "p12_p18_p19.zip"
    ]
    
    loaded_state_dicts = []
    
    print(f"Consolidating {len(model_files)} models from {model_dir}...")
    
    for filename in model_files:
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {filename} not found, skipping.")
            continue
            
        print(f"Loading {filename}...")
        try:
            ckpt = torch.load(path, map_location='cpu')
            # Handle both state_dict and full checkpoint dict
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
            
            loaded_state_dicts.append(state_dict)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    if not loaded_state_dicts:
        print("No models loaded. Consolidation failed.")
        return

    # Initialize averaged state_dict with the first one
    avg_state_dict = OrderedDict()
    keys = loaded_state_dicts[0].keys()
    
    for key in keys:
        # Sum up weights for this key across all models
        stack = torch.stack([sd[key] for sd in loaded_state_dicts])
        avg_state_dict[key] = torch.mean(stack.float(), dim=0)
        
    # Save the consolidated model
    torch.save(avg_state_dict, output_path)
    print(f"Successfully consolidated {len(loaded_state_dicts)} models into {output_path}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    output_model = os.path.join(current_dir, "final_consolidated_model.pth")
    consolidate_models(current_dir, output_model)
