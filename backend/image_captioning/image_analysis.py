from PIL import Image
import io, base64, torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from flask import jsonify



def analyze_image(image_bytes, processor, model, prompt="Describe the image in a single sentence."):
    
    img_bytes = image_bytes

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((256, 256))
        print(image.size)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # Build a multimodal prompt

    # Qwen2-VL supports chat formatting with images
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]
    print("prcessor", processor)
    # Preprocess and generate


    chat_prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
    )

    inputs = processor(
    text=chat_prompt,
    images=[image],
    padding=True,
    return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            top_p=0.9,
            temperature=0.6
        )

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Optional: strip the chat preamble if present
    # Many chat templates echo the prompt; keep last turn:
    answer = text.split("assistant\n")[-1].strip() if "assistant" in text else text.strip()

    return answer