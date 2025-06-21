import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense


class main:

    def handle_missing_values(df):
        # Fill in all missing values with 'Unknown'
        df = df.replace(["", " ", "\t", "\n"], np.nan)
        print(pd.isnull(df).sum())
        df.fillna('Unknown')
        print(pd.isnull(df).sum())

    def summary_statistics(df):
        # Show the statisitcs for the dataset
        df.describe()
        print(df.shape)
    
    def highest_rated_recipes(df):
        # Sort the rating_avg column is descending order
        top_rated = df.sort_values('rating_avg', ascending=False).head(10)
        print(top_rated[['title']])
    

    def visualise(df):
        # Create a scatter plot of the rating_avg and rating_val where all rating_avg are rounded to 1 d.p.
        plt.scatter(df['rating_avg'].round(1), df['rating_val'])
        plt.xlabel("Average Ratings")
        plt.ylabel("Number Of Ratings")
        plt.show()

        # Calculate threshold - In this case it is 25% of the rating_val mean.
        rating_val_mean = df['rating_val'].mean()
        threshold = rating_val_mean * 0.25
        print(threshold)

        '''By finding the average we can determine that 25% of average is suitable 
        because it takes into account the variability in the data while still ensuring 
        that the average rating is based on a sufficient number of ratings to be reliable.'''

    def combine_features(df):
        features = ['title','rating_avg','rating_val','total_time','category','cuisine', 'ingredients']
        df['combine_features'] = df[features].astype(str).apply(' '.join, axis=1)

    def generate_cosine_similarity_matrix(df):
        # Create a CountVectorizer object and fit_transform the text data
        count = CountVectorizer()
        count_matrix = count.fit_transform(df['combine_features'])
        
        # Calculate the cosine similarity matrix of the count_matrix
        cosine_sim = cosine_similarity(count_matrix)
        
        # Print the cosine similarity matrix
        print(cosine_sim)

    def chicken_and_coconut_curry(df):
        # Create a CountVectorizer object and fit_transform the text data
        count = CountVectorizer()
        count_matrix = count.fit_transform(df['combine_features'])

        # Get the index of the recipe 'Chicken and coconut curry' and its corresponding vector
        recipe_index = df[df['title'] == 'Chicken and coconut curry'].index[0]
        chicken_coconut_curry_vector = count_matrix[recipe_index].toarray()[0]

        # Calculate the cosine similarity between 'Chicken and coconut curry' and all other recipes
        similarity_scores = []
        for i in range(count_matrix.shape[0]):
            if i == recipe_index:
                continue
            recipe_vector = count_matrix[i].toarray()[0]
            scalar_product = np.dot(chicken_coconut_curry_vector, recipe_vector)
            norm1 = np.linalg.norm(chicken_coconut_curry_vector)
            norm2 = np.linalg.norm(recipe_vector)
            cosine_similarity = scalar_product / (norm1 * norm2)
            similarity_scores.append((i, cosine_similarity))

        # Get the top 10 similar recipes and print their titles
        similar_recipes = sorted(similarity_scores, key=lambda x:x[1], reverse=True)
        similar_recipes = similar_recipes[:10]
        recommended_recipes = [df.iloc[i[0]]['title'] for i in similar_recipes]
        print("The first 10 recipe recommendations for a user who has liked the recipe 'Chicken and coconut curry' are:")
        for recipe in recommended_recipes:
            print(recipe)

    def vec_space_method(recipe, df):
        count = CountVectorizer()

        features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']
        
        # Combine the text columns
        text_columns = ['title', 'ingredients', 'category', 'cuisine']
        df['combine_text'] = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # Fit the CountVectorizer object and transform the data
        count_matrix = count.fit_transform(df['combine_text'])

        # Create a matrix for the numerical columns
        num_columns = ['total_time', 'rating_avg', 'rating_val']
        num_matrix = df[num_columns].to_numpy()
        
        # Combine the text and numerical features
        combined_matrix = sparse.hstack((count_matrix, num_matrix)).tocsr()
        
        # Get the index of the input recipe
        recipe_index = df[df['title'] == recipe].index[0]

        # Calculate cosine similarity between the input recipe and all other recipes
        similarity_scores = []
        for i in range(combined_matrix.shape[0]):
            if i == recipe_index:
                continue
            recipe_vector = combined_matrix[i].toarray()[0]
            similar_recipe_vector = combined_matrix[recipe_index].toarray()[0]
            scalar_product = np.dot(similar_recipe_vector, recipe_vector)
            norm1 = np.linalg.norm(similar_recipe_vector)
            norm2 = np.linalg.norm(recipe_vector)
            cosine_similarity = scalar_product / (norm1 * norm2)
            similarity_scores.append((i, cosine_similarity))

        # Sort the similarity scores and get the top 10 similar recipes
        similar_recipes = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
        similar_recipe_indices = [i[0] for i in similar_recipes]
        similar_recipes = df.iloc[similar_recipe_indices]['title'].tolist()

        # Print the top 10 similar recipes
        print("10 similar recipes using the vector-space method: ")
        for recipe in similar_recipes:
            print(recipe)

    def knn_similarity(recipe, df):

        # Create a CountVectorizer object
        count = CountVectorizer()

        # Combine the text columns
        text_columns = ['title', 'ingredients', 'category', 'cuisine']
        df['combine_text'] = df[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # Fit the CountVectorizer object and transform the data
        count_matrix = count.fit_transform(df['combine_text'])

        # Create a matrix for the numerical columns
        num_columns = ['total_time', 'rating_avg', 'rating_val']
        num_matrix = df[num_columns].to_numpy()

        # Combine the count matrix and the numerical matrix
        combined_matrix = sparse.hstack((count_matrix, num_matrix))

        # Get the index of the given recipe
        recipe_index = df[df['title'] == recipe].index[0]

        # Train a KNN model on the datasetclear
        knn_model = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
        knn_model.fit(combined_matrix)

        # Find the indices of the 10 most similar recipes to the given recipe
        _, indices = knn_model.kneighbors(combined_matrix.getrow(recipe_index))

        # Get the titles of the top 10 similar recipes
        similar_recipe_indices = indices[0][1:]
        similar_recipes = df.iloc[similar_recipe_indices]['title'].tolist()

        print("10 similar recipes using the KNN model: ")
        for recipe in similar_recipes:
            print(recipe)

            
       
    # def ANN_algorithm(df):
        # # Select the significant features
        # X = ?
        # y = np.where(df['rating_avg'] > 4.2, 1, 0)

        # # Define the ANN model
        # model = Sequential()
        # model.add(Dense(units=10, input_dim=5, activation='relu'))
        # model.add(Dense(units=6, activation='relu'))
        # model.add(Dense(units=1, activation='sigmoid'))

        # # Compile the model
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # # Train the model
        # model.fit(X, y, epochs=50, batch_size=32)
        # predictions = np.round(model.predict(X))
        # _, accuracy = model.evaluate(X, y)
        # print('Accuracy: %.2f' % (accuracy*100))

        # # summarize the first 5 cases
        # for i in range(5):
        #     print('predicted %d (actual %d)' % (predictions[i], y[i]))

class test:

    # TASK 1 #

    df = pd.read_csv("recipes.csv")
    # main.handle_missing_values(df)
    # main.summary_statistics(df)
    # main.highest_rated_recipes(df)

    # TASK 2 #

    # main.visualise(df)
    # NEEDS COMPLETING

    # TASK 3 #

    ' ---> main.combine_features(df) <--- IS REQUIRED FOR OTHER FUNCTIONS TO WORK!'
    main.combine_features(df)
    # main.generate_cosine_similarity_matrix(df)
    main.chicken_and_coconut_curry(df)

    # TASK 4 #

    # main.vec_space_method('Chicken and coconut curry', df)

    # TASK 5 #

    # main.knn_similarity('Chicken and coconut curry', df)

    # TASK 6 #

    # TEST CASES #

    # • User 1 likes ‘Chicken tikka masala’
    # • User 2 likes ‘Albanian baked lamb with rice’
    # • User 3 likes ‘Baked salmon with chorizo rice’
    # • User 4 likes ‘Almond lentil stew’ 

    # main.vec_space_method('Chicken tikka masala', df)
    # main.knn_similarity('Chicken tikka masala', df)

    '''Note: An assumption has been made here that 'Albanian baked lamb with rice' refers to
    'Albanian baked lamb with rice (Tavë kosi)' as the current system cannot detect partial titles.'''

    # main.vec_space_method('Albanian baked lamb with rice (Tavë kosi)', df)
    # main.knn_similarity('Albanian baked lamb with rice (Tavë kosi)', df)

    # main.vec_space_method('Baked salmon with chorizo rice', df)
    # main.knn_similarity('Baked salmon with chorizo rice', df)

    # main.vec_space_method('Almond lentil stew', df)
    # main.knn_similarity('Almond lentil stew', df)

    # EVALUATION #

    '''The system currently recommends 10 recipies (for both KNN and vector-space) to the user
    so we can define coverage as (recommended recipes / total recipes in the dataset). Both 
    algorithms will have the same coverage -> (10/3293) * 100 = 0.30%. This may be considered
    as a low coverage as both systems are recommending a very small fraction of the total 
    available items.'''

    '''Personalisation measures the dissimilarity between the recommended recipies and the 
    user test set provided.'''

    # VECTOR-SPACE COSINE SIM MATRIX BETWEEN USERS 1 - 4 # 

    '''[[1.         0.27382078 0.55666994 0.11513282]
        [0.27382078 1.         0.25041709 0.31234752]
        [0.55666994 0.25041709 1.         0.12635079]
        [0.11513282 0.31234752 0.12635079 1.        ]]'''
    
    '''The average cosine similarity of the upper triangle is: 0.27245649. 

    1 - 0.27245649 = 0.7275
    
    This personalisation score shows the users are recommended different items and therefore has a 
    high personalisation.'''

    # KNN Algorithm COSINE SIM MATRIX BETWEEN USERS 1 - 4 #

    '''[[1.         0.16187405 0.42492854 0.06937459]
        [0.16187405 1.         0.18015094 0.08823529]
        [0.42492854 0.18015094 1.         0.04503773]
        [0.06937459 0.08823529 0.04503773 1.        ]]'''
    
    '''The average cosine similarity of the upper triangle is: 0.16160019
    
    1 - 0.16160019 = 0.83839981
    
    This perosnalisation score for the KNN algorithm indicates a higher dissimilarity thus providing
    more personalised recipes for the user.'''
    
    # TASK 7 #

    
    # main.ANN_algorithm(df)

