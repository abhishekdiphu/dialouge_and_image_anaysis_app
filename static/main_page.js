async function RunSentimentAnalysis() {
    const text = document.getElementById("textToAnalyze").value.trim();
    const responseDiv = document.getElementById("system_response");

    // 🟡 Step 1: Input validation
    if (!text) {
        responseDiv.innerHTML = `
            <div class="alert alert-warning text-center" role="alert">
                ⚠️ Please enter some text before analyzing.
            </div>
        `;
       
       
        return;
    }

    if (/^\d+(\s*\d+)*$/.test(text)) {
        responseDiv.innerHTML = `
            <div class="alert alert-warning text-center" role="alert">
                ⚠️ Numbers are not allowed. Please enter valid text.
            </div>
        `;
        return;
    }

    if (/^[^a-zA-Z0-9]+$/.test(text)) {
        responseDiv.innerHTML = `
            <div class="alert alert-warning text-center" role="alert">
                ⚠️ Symbols are not allowed. Please enter valid words or sentences.
            </div>
        `;
        return;
    }

     if (!/[a-zA-Z]/.test(text) && /[0-9]/.test(text) && /[^a-zA-Z0-9\s]/.test(text)){
         responseDiv.innerHTML = `
            <div class="alert alert-warning text-center" role="alert">
                ⚠️ Symbols with numbers are not allowed. Please enter valid words or sentences.
            </div>
        `;
        return;

     }

    // 🟢 Step 2: Show loading animation
    responseDiv.innerHTML = `
        <div class="text-center mt-3">
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-2">Analyzing sentiment with BERT...</p>
        </div>
    `;

    // 🧠 Step 3: Send text to Flask backend
    try {
        const res = await fetch(`/sentiment_analysis?textToAnalyze=${encodeURIComponent(text)}`);
        
        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        // 🔹 Step 4: Get the response (adjust based on backend format)
        // If Flask returns JSON (recommended)
        const data = await res.json();

        if (data.error) {
            responseDiv.innerHTML = `
                <div class="alert alert-danger text-center" role="alert">
                    ❌ ${data.error}
                </div>
            `;
            return;
        }

        // 🟣 Step 5: Choose alert style based on label
        const sentiment = data.label.toUpperCase();
        const score = (data.score * 100).toFixed(2);

        const alertType =
            sentiment === "POSITIVE" ? "success" :
            sentiment === "NEGATIVE" ? "danger" : "secondary";

        // 🟩 Step 6: Display the real model output
        responseDiv.innerHTML = `
            <div class="alert alert-${alertType} text-center">
                <h4><strong>Sentiment:</strong> ${sentiment}</h4>
                <p><strong>Confidence:</strong> ${score}%</p>
            </div>
        `;
    } catch (err) {
        // 🔴 Step 7: Handle errors
        responseDiv.innerHTML = `
            <div class="alert alert-danger text-center" role="alert">
                ❌ Could not connect to Flask server.<br>
                <small>${err.message}</small>
            </div>
        `;
    }
}



async function initImageAnalyzer() {
    const imgInput = document.getElementById("img");
    const promptInput = document.getElementById("prompt");
    const btn = document.getElementById("go");
    const out = document.getElementById("out");
    const preview = document.getElementById("preview");
    const previewWrap = document.getElementById("previewWrap");


    let fileBlob = null;

    // Disable Dialogue Summarization section until image explanation is done
    const dialogueInput = document.getElementById("dialogueInput");
    const dialogueButton = document.getElementById("dialogueButton");

    dialogueInput.disabled = true;
    dialogueButton.disabled = true;

    // When the user selects an image
    imgInput.addEventListener("change", () => {
        const f = imgInput.files?.[0];
        fileBlob = f || null;
        btn.disabled = !fileBlob;

        if (fileBlob) {
            const url = URL.createObjectURL(fileBlob);
            preview.src = url;
            previewWrap.style.display = "block";
        } else {
            previewWrap.style.display = "none";
        }
    });

    // When the user clicks "Explain"
    btn.addEventListener("click", async () => {
        if (!fileBlob) return;

        // Lock dialogue summarization while analyzing image
        dialogueInput.disabled = true;
        dialogueButton.disabled = true;

        out.innerHTML = `
            <div class="text-center">
                <div class="spinner-border" role="status"></div>
                <p class="mt-2">Analyzing image...</p>
            </div>
        `;

        const form = new FormData();
        form.append("image", fileBlob);
        form.append("prompt", promptInput.value || "Explain the image in detail.");

        try {
            // ✅ Call Flask backend
            const res = await fetch("/image_analysis", {
                method: "POST",
                body: form
            });

            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);

            out.innerHTML = `
                <div class="alert alert-info" style="white-space: pre-wrap">
                    ${data.answer}
                </div>
            `;
            // Unlock dialogue summarization after successful image result
            dialogueInput.disabled = false;
            dialogueButton.disabled = false;
        } catch (e) {
            out.innerHTML = `
                <div class="alert alert-danger">
                    Error: ${e.message}
                </div>
            `;
        }
    });
}


  async function RunDialogSummarization() {
    const inputText = document.getElementById("dialogueInput").value.trim();
    const outputDiv = document.getElementById("dialogue_summary_output");

    inputText.disabled = true;
    outputDiv.disabled = true;

    // Clear previous output
    outputDiv.innerHTML = "";

    // Validate input
    if (!inputText) {
      outputDiv.innerHTML = "<span class='text-danger'>⚠️ Please enter some text to summarize.</span>";
      return;
    }

    // Show loading state
    outputDiv.innerHTML = "<em>⏳ Summarizing document...</em>";

    try {
      // Send request to backend (adjust URL if needed)
      const response = await fetch("/dialog_summerization", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dialogue: inputText })   // 'document' key expected by backend
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      // Parse JSON response
      const data = await response.json();

      // Display summary
      if (data.summary) {
        outputDiv.innerHTML = `<strong>${data.summary}</strong>`;
      } else {
        outputDiv.innerHTML = "<span class='text-warning'>⚠️ No summary returned from server.</span>";
      }

    } catch (error) {
      // Handle any errors
      console.error(error);
      outputDiv.innerHTML = `<span class='text-danger'>❌ Error: ${error.message}</span>`;
    }
  }



// Automatically initialize once DOM is ready
document.addEventListener("DOMContentLoaded", initImageAnalyzer);



// 🌈 Optional: animated background transitions
const gradients = [
    "linear-gradient(135deg, #667eea, #764ba2)",
    "linear-gradient(135deg, #36d1dc, #5b86e5)",
    "linear-gradient(135deg, #00c9ff, #92fe9d)",
    "linear-gradient(135deg, #ff9966, #ff5e62)",
    "linear-gradient(135deg, #f7971e, #ffd200)"
];

let i = 0;
setInterval(() => {
    document.body.style.transition = "background 2s ease";
    document.body.style.background = gradients[i];
    i = (i + 1) % gradients.length;
}, 8000);
