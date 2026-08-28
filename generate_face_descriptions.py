import os
import sys
import glob
import torch
import time
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
# ================= Command-Line Argument Handling =================
if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <rgb_dir> <parsing_map_dir> <output_dir>")
    print(f"Example: python {sys.argv[0]} ./images_rgb ./images_parsing ./output_captions")
    sys.exit(1)

rgb_dir = sys.argv[1]
map_dir = sys.argv[2]
output_dir = sys.argv[3]
os.makedirs(output_dir, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
rgb_files = sorted([p for p in glob.glob(os.path.join(rgb_dir, "*"))
                    if p.lower().endswith(IMG_EXTS)])

if not rgb_files:
    print(f"No images found in {rgb_dir}")
    sys.exit(0)

# ================= Model Loading =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
print(f"Loading model: {MODEL_ID}...")

try:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,  # Use FP16 for faster inference.
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
except Exception:
    print("Flash Attention 2 not available, fallback to default...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()
torch.set_grad_enabled(False)  # Disable gradient computation.
# ================= System Prompt Definition =================
SYSTEM_PROMPT = """You are an expert visual analyst for high-fidelity face restoration, specialized in generating dense, natural language descriptions for T5 text encoders.

You will receive two images:
1. **Image 1 (RGB)**: Source for color, texture, lighting, and fine details.
2. **Image 2 (Parsing Map)**: Source for GEOMETRY (shapes, open/closed states, relative sizes). This is the ground truth for structure.

### TASK:
Generate a **single, coherent, and objective natural language paragraph** describing the person. 
The description must be grammatically correct, using prepositions (e.g., "with", "featuring", "under") to connect features naturally.

### STRICT WRITING RULES:
1.  **NO Metaphors or Subjective Language**: Do NOT use phrases like "soulful eyes," "radiant glow," or "kissed by the sun." Stick strictly to observational adjectives (e.g., "bright green," "smooth," "soft lighting").
2.  **Logic Flow**: Start with the global appearance and lighting, then move to specific facial features (Face -> Eyes/Brows -> Nose -> Mouth).
3.  **Map Interpretation**: The Parsing Map uses artificial colors to label regions. **IGNORE the colors of the map itself.** Only use the map to determine the SHAPE and STATE (open/closed) of features. Use the RGB image for the actual skin/hair/eye colors.
4.  **Source Fidelity**: You must strictly adhere to the Geometry from the Map and Details from RGB as defined below.

### ATTRIBUTE INTEGRATION GUIDE:

**1. The Overall Appearance & Accessories:**
   - **Source**: RGB (Content & Lighting).
   - **Combine**: Age, gender, race, skin tone, hair (color/style), pose, expression, accessories and lighting quality.
   - **Draft**: "A [age] [race] [gender] with [skin_tone] skin and [hair_style] [hair_color] hair is [pose/expression] under [lighting_quality] lighting. [He/She] is wearing [accessories]."
   *(Note: If no accessories are visible, omit the second sentence.)*
   

**2. Face & Skin Details:**
   - **Source**: Map (Shape) + RGB (Texture/Details).
   - **Combine**: Face shape, wrinkles, moles, freckles, makeup_style.
   - **Draft**: "[He/She] features a [shape] face with [wrinkles/moles/freckles] and a [makeup_style] finish."

**3. Eyes & Eyebrows:**
   - **Source**: Map (Shape/State) + RGB (Color/Lashes/Grooming).
   - **Combine**: Eye shape/state, iris color, eyelashes + Eyebrow shape/density.
   - **Draft**: "[His/Her] [shape], [state] eyes are [color] with [lash details], framed by [shape], [density] eyebrows."

**4. Nose & Mouth:**
   - **Source**: Map (Shape/Size/State) + RGB (Texture/Color).
   - **Combine**: Nose shape/Details + Mouth size/state, lip texture/color, teeth visibility.
   - **Draft**: "[He/She] has a [shape] nose with [Details]. [His/Her] [size] mouth is [state], revealing [lip details] and [teeth visibility]."

### ENUM REFERENCE (Use these to ground your vocabulary):
- **Age**: teen/young/middle-aged/senior
- **Race**: asian/indian/black/white/middle_eastern/latino_hispanic
- **Emotion**: neutral/smiling/sad/angry/surprised/disgusted/fearful
- **Pose**: facing front/left/right
- **Accessories**: necklace/hat/necktie/earrings/eyeglasses/none
- **Geometry**: narrow/round/sharp/arched/thin/thick
- **Details**: dry/glossy/smooth/cracked/pale/dark

### OUTPUT FORMAT:
Return ONLY the raw paragraph text. Do not use prefixes like `[Global]:` or bullet points.

**Example of Desired Output:**
"A young Asian female with pale skin and long black hair with bangs is smiling under soft natural lighting. She is wearing circular gold earrings and a silver necklace. She features an oval face shape with a heavy makeup finish and no visible freckles. Her narrow, open eyes are dark brown with thick eyelashes, complemented by arched, neatly groomed black eyebrows. The center of the face shows a small nose with smooth skin texture. Her mouth is slightly open, displaying thick, glossy red lips with upper teeth visible."
"""

# ================= Batch Parameters =================
BATCH_SIZE =100 # Adjust according to available GPU memory.

print(f"Starting processing of {len(rgb_files)} images in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(rgb_files), BATCH_SIZE), desc="Processing batches", unit="batch"):
    batch_files = rgb_files[i:i+BATCH_SIZE]
    batch_messages = []
    batch_basenames = []
    
    for rgb_path in batch_files:
        filename = os.path.basename(rgb_path)
        basename = os.path.splitext(filename)[0]
        batch_basenames.append(basename)

        # Find the corresponding parsing map.
        map_path = None
        for ext in IMG_EXTS:
            candidate = os.path.join(map_dir, basename + ext)
            if os.path.exists(candidate):
                map_path = candidate
                break
        if not map_path:
            tqdm.write(f"Skipping {filename}: Map not found.")
            continue

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": rgb_path},
                {"type": "image", "image": map_path},
                {"type": "text", "text": "Here are Image 1 (RGB) and Image 2 (Parsing Map). Please generate the description strictly following the system rules."}
            ]}
        ]
        batch_messages.append(messages)

    if not batch_messages:
        continue

    try:
        # Preprocess.
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        images_list = [process_vision_info(msg)[0] for msg in batch_messages]  # Use only image_inputs.
        inputs = processor(
            text=texts,
            images=images_list,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate.
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

        # Decode.
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Save.
        for basename, output_text in zip(batch_basenames, outputs):
            txt_out_path = os.path.join(output_dir, basename + ".txt")
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(output_text)

        tqdm.write(f"Processed batch {i//BATCH_SIZE + 1} / {len(rgb_files)//BATCH_SIZE + 1}")

    except Exception as e:
        tqdm.write(f"Error processing batch starting with {batch_files[0]}: {e}")
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()

print(f"\nDone! Results saved to {output_dir}")
