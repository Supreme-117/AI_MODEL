# from transformers import pipeline


# classifier = pipeline(
#     task="text-classification",
#     model=r"/Complaint",
#     tokenizer=r"/Complaint"
# )

# label_map = {
#     "LABEL_0": "Road",
#     "LABEL_1": "Sanitation",
#     "LABEL_2": "Electricity",
#     "LABEL_3": "Water"
# }

# print(label_map[classifier("Pani ka ki bucket fill nahi hota")[0]['label']])
# print(label_map[classifier("There is a pani leakage near my house")[0]['label']]) 


from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)


classifier = pipeline("text-classification", model="/Complaint", tokenizer="/Complaint")
label_map = {
    "LABEL_0": "Road",
    "LABEL_1": "Sanitation",
    "LABEL_2": "Electricity",
    "LABEL_3": "Water"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    complaint_text = data.get("complaint", "")
    result = classifier(complaint_text)[0]
    return jsonify({
        "complaint": complaint_text,
        "predicted_label": result["label"],
        "confidence": float(result["score"])
    })

@app.route("/predict_get", methods=["GET"])
def predict_get():
    complaint_text = request.args.get("complaint", "")
    if not complaint_text:
        return jsonify({"error": "Please provide a complaint, e.g., /predict_get?complaint=pani+nahi+aa+raha"})
    result = classifier(complaint_text)[0]
    return jsonify({
        "complaint": complaint_text,
        "predicted_label": label_map[result["label"]],
        "confidence": float(result["score"])
    })

if __name__ == "__main__":
    app.run(debug=True)
