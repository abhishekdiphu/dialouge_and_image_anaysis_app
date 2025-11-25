
# Import Flask, render_template, request from the flask pramework package : TODO
# Import the sentiment_analyzer function from the package created: TODO
from flask import Flask, render_template, request, jsonify
from sentiment_analysis.sentiment_analysis import analyze_text
from image_captioning.image_analysis import analyze_image
from text_summerization.summerizaton import summerize_text

from PIL import Image
import io, base64, torch
from transformers import AutoProcessor, AutoModelForVision2Seq


# zip longest so human/ai alternate correctly
from itertools import zip_longest





print("Model successfully loaded!")

data_for_analysis = {"human" : [], "ai" : []}




app = Flask("Sentiment Analyzer")

@app.route("/sentiment_analysis", methods=["GET"])
def sent_analyzer():
    ''' This code receives the text from the HTML interface and 
        runs sentiment analysis over it using sentiment_analysis()
        function. The output returned shows the label and its confidence 
        score for the provided text.
    '''
    text = request.args.get('textToAnalyze', '')
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    result = analyze_text(text)
    return jsonify(result)








@app.route("/image_analysis", methods=["POST"])
def sent_to_image_analyzer():
    
    """
    Expects multipart/form-data with:
      - image: file
      - prompt: text (e.g. "Explain this image in detail.")
    Returns JSON: { "answer": "..." }
    """
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


    device = "mps"      # Apple Silicon GPU

    print(f"Using device: {device}")

    # ✅ Load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)


    dtype = torch.float16  # works fine on Apple Silicon


    # ✅ Load model (disable device_map on MPS/CPU)
    model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
    )

    # ✅ Move model to the right device
    model.to(device)
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    prompt = request.form.get("prompt", "Explain the image in detail.")
    f = request.files["image"]
    img_bytes = f.read()
    print("the prompt passed with the image is ",prompt)

    
    answer = analyze_image(img_bytes, processor, model, prompt)

    data_for_analysis["human"].append(str(prompt))
    data_for_analysis["ai"].append(str(answer))
    
    return jsonify({"answer": answer})


@app.route("/dialog_summerization", methods=["POST"])
def send_to_summerization():

    print("dialog_summerization endpoint reached")
    
    """Receive dialogue from frontend and return summary"""

    instruction = "Summarize the following conversation: "
    data = request.get_json(silent=True) or {}
    print("Received data:", data)
    text = data.get("dialogue") or data.get("textToAnalyze", "")
    print("text recived", text)
    data_for_analysis["human"].append(str(text))
    
    output_parts = []
    for h, a in zip_longest(data_for_analysis['human'], data_for_analysis['ai']):
        if h:
            output_parts.append(f"#Human#: {h}.")
        if a:
            output_parts.append(f"#AI#: {a}.")

    final_text = "\n".join(output_parts).strip()

    print("final text: ", final_text)


    if not text or not text.strip():
        return jsonify({"error": "No dialogue text provided."}), 400

    
    try:
        

        result = summerize_text(final_text)


        return result
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/")
def render_index_page():
    ''' This function initiates the rendering of the main application
        page over the Flask channel
    '''
    return render_template('index.html')




if __name__ == "__main__":

    
    app.run(host="0.0.0.0", port=9000)
