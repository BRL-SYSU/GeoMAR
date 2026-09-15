import argparse
import os
import torch
from transformers import T5Tokenizer, T5EncoderModel

# Set paths from command-line arguments
parser = argparse.ArgumentParser(description="Encode text descriptions into T5 features.")
parser.add_argument("--text-folder", required=True, help="Folder containing text descriptions")
parser.add_argument("--output-folder", required=True, help="Folder for output text features")
args = parser.parse_args()

TEXT_FOLDER = args.text_folder
OUTPUT_FOLDER = args.output_folder
if not os.path.isdir(TEXT_FOLDER):
    parser.error(f"Text folder does not exist: {TEXT_FOLDER}")
TEXT_FEATURES_FOLDER = os.path.join(OUTPUT_FOLDER)
os.makedirs(TEXT_FEATURES_FOLDER, exist_ok=True)

# Load the T5-Large model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "t5-large"  # Use the T5-Large model
tokenizer = T5Tokenizer.from_pretrained(model_name)
text_encoder = T5EncoderModel.from_pretrained(model_name).to(device)

# Set the target dimension
TARGET_TEXT_DIM = 1024  # T5-Large output dimension

def get_text_features(text):
    """Extract text features with T5-Large."""
    try:
        # Tokenize and encode
        encoded_input = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            model_output = text_encoder(**encoded_input)
            text_features = model_output.last_hidden_state  # Preserve sequence features
            return text_features  # [batch_size, seq_len, hidden_dim]

    except Exception as e:
        print(f"Error processing text: {e}")
        return None

# Process all text files
processed_count = 0
skipped_count = 0
error_count = 0

print("Starting to process text files...")


for file_name in os.listdir(TEXT_FOLDER):
    if not file_name.endswith('.txt'):
        continue

    file_path = os.path.join(TEXT_FOLDER, file_name)
    base_name = os.path.splitext(file_name)[0]
    output_file = os.path.join(TEXT_FEATURES_FOLDER, f"{base_name}_text.pt")

    # Check whether the output file already exists
    if os.path.exists(output_file):
        print(f"Skipping {file_name}: already processed")
        skipped_count += 1
        continue

    try:
        print(f"Processing: {file_path}")

        # Read the text description
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Extract text features
        text_features = get_text_features(text)

        if text_features is None:
            print(f"Could not extract features from {file_name}; skipping")
            error_count += 1
            continue

        # Save text features
        torch.save(text_features, output_file)
        print(f"Saved text features to: {output_file}")
        processed_count += 1

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        error_count += 1

print(f"Text processing complete! Processed: {processed_count}, skipped: {skipped_count}, errors: {error_count}")
