import pandas as pd
import joblib

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

real_news = true_df.iloc[0]["text"]
fake_news = fake_df.iloc[0]["text"]

X = vectorizer.transform([real_news, fake_news])

predictions = model.predict(X)

print("Real article prediction:", predictions[0])
print("Fake article prediction:", predictions[1])