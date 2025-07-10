import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset and train model
df = pd.read_csv("data/heart.csv")
X = df.drop("condition", axis=1)
y = df["condition"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Create main window
app = tk.Tk()
app.title("Heart Disease Predictor")
app.geometry("400x600")

features = list(X.columns)
entries = {}

# Create label and entry for each feature
for i, feature in enumerate(features):
    label = tk.Label(app, text=feature)
    label.grid(row=i, column=0, padx=10, pady=5, sticky='w')
    entry = tk.Entry(app)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[feature] = entry

# Prediction function
def predict():
    try:
        values = [float(entries[feature].get()) for feature in features]
        pred = model.predict([values])[0]
        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Predict button
predict_btn = tk.Button(app, text="Predict", command=predict, bg="blue", fg="white")
predict_btn.grid(row=len(features), column=0, columnspan=2, pady=20)

app.mainloop()