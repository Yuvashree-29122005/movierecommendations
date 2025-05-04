import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets (Example format)
ratings = pd.read_csv('ratings.csv')     # userId, movieId, rating
movies = pd.read_csv('movies.csv')       # movieId, title, genres
users = pd.read_csv('users.csv')         # userId, age, gender, occupation, etc.

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')
data = pd.merge(data, users, on='userId')

# Encode genres (multi-label one-hot)
data['genres'] = data['genres'].apply(lambda g: g.split('|'))
genres = data['genres'].explode().unique()
for genre in genres:
    data[genre] = data['genres'].apply(lambda x: int(genre in x))

# Preprocess user features
user_features = ['age', 'gender', 'occupation']
data['gender'] = LabelEncoder().fit_transform(data['gender'])  # e.g., Male=1, Female=0

# Aggregate user-movie features
user_movie = data.groupby(['userId', 'movieId']).agg({
    'rating': 'mean',
    'age': 'first',
    'gender': 'first',
    'occupation': 'first',
    **{genre: 'mean' for genre in genres}
}).reset_index()

# Feature selection
features = ['age', 'gender', 'occupation'] + list(genres)
X = user_movie[features]
y = (user_movie['rating'] >= 3.5).astype(int)  # Binary: 1 = liked, 0 = not liked

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------- CLUSTERING: Group Similar Users --------
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
user_movie['cluster'] = clusters

# -------- CLASSIFICATION: Predict Preferences --------
X_train, X_test, y_train, y_test = train_test_split(
    np.hstack((X_scaled, clusters.reshape(-1, 1))), y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, predictions))

# -------- RECOMMENDATION: For a New User --------
def recommend_for_user(new_user_profile):
    """
    new_user_profile: dict with age, gender, occupation, genres
    Example: {'age': 25, 'gender': 'Male', 'occupation': 4, 'Action': 1, 'Comedy': 0, ...}
    """
    # Prepare feature vector
    profile_df = pd.DataFrame([new_user_profile])
    profile_df['gender'] = LabelEncoder().fit_transform(profile_df['gender'])
    profile_scaled = scaler.transform(profile_df[features])

    # Predict cluster and preference
    cluster_label = kmeans.predict(profile_scaled)[0]
    profile_with_cluster = np.hstack((profile_scaled, [[cluster_label]]))
    liked = clf.predict_proba(profile_with_cluster)[0][1]

    if liked > 0.6:
        return f"Recommended movies based on similar users in cluster {cluster_label}!"
    else:
        return "No strong preferences detected. Consider more inputs."

# Example usage
print(recommend_for_user({
    'age': 22, 'gender': 'Male', 'occupation': 3,
    'Action': 1, 'Comedy': 1, 'Drama': 0, 'Romance': 0, 'Horror': 0,
    'Adventure': 1, 'Sci-Fi': 1, 'Thriller': 0, 'Fantasy': 0
}))
