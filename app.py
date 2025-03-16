from flask import Flask, render_template, request
import pickle
import numpy as np
app=Flask(__name__)

top_50_books=pickle.load(open('top_50_books.pkl','rb'))
final = pickle.load(open('final.pkl', 'rb'))  # Load 'final' dataframe
model = pickle.load(open('model.pkl', 'rb'))  # Load trained model
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))  # Load pivot table

books = final[['title', 'img_url']].drop_duplicates()
@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(top_50_books['title'].values),
                           image=list(top_50_books['img_url'].values),
                           votes=list(top_50_books['num_ratings'].values)
                           )

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_ui():
    if request.method == 'POST':
        book_name = request.form['book_name']
        recommended_books = recommend_book(book_name)
        return render_template('recommend.html', books=recommended_books)
    return render_template('recommend.html', books=[])



def recommend_book(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        _, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        recommended_books = []
        for i in range(len(suggestion)):
            books_list = book_pivot.index[suggestion[i]]
            for j in books_list:
                img_url = books[books['title'] == j]['img_url'].values[0]  # Fetch Image URL
                recommended_books.append((j, img_url))

        return recommended_books
    except IndexError:
        return [("Book not found", "https://via.placeholder.com/150")]

if __name__=='__main__':
    app.run(debug=True)