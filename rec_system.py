import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

movies = pd.read_csv("movies.csv")
movies["clean_title"] = movies["title"].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x))
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])


def search(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results


ratings = pd.read_csv("ratings.csv")


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    same_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    same_user_recs = same_user_recs.value_counts() / len(similar_users)

    same_user_recs = same_user_recs[same_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(same_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([same_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


while True:
    title = input("Movie Title: ")
    if len(title) > 5:
        results = search(title)
        movie_id = results.iloc[0]["movieId"]
        print(find_similar_movies(movie_id))
    else:
        break
