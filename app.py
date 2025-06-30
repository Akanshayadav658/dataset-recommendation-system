from flask import Flask, request, render_template
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load saved model and data
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = joblib.load('dataset.pkl')  # pandas DataFrame

@app.route('/', methods=['GET', 'POST'])
def recommend():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        query_vec = vectorizer.transform([query])
        distances, indices = model.kneighbors(query_vec)

        for i in indices[0]:
            result = {
                'name': df.iloc[i]['Dataset_name'],
                'author': df.iloc[i]['Author_name'],
                'files': df.iloc[i]['No_of_files'],
                'size': df.iloc[i]['size'],
                'type': df.iloc[i]['Type_of_file'],
                'upvotes': df.iloc[i]['Upvotes'],
                'medals': df.iloc[i]['Medals'],
                'usability': df.iloc[i]['Usability'],
                'link': df.iloc[i]['Dataset_link']
            }
            results.append(result)

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
