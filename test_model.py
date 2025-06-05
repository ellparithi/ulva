from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-uncased")
result = qa({
    "question": "What is the treatment for flu?",
    "context": "Flu is commonly treated with rest, fluids, and over-the-counter medication."
})
print(result)
