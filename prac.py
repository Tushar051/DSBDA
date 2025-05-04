# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv('FB_data.csv')
# print(df.head())


# # Data Cleaning

# print(df.isnull().sum())
# df = df.fillna(0)
# df = df.drop_duplicates()
# print(df.shape)

# #Data Integration

# # df2 = pd.read_csv('')
# # merge_csv = pd.merge(df, df2, on='post_id', how='inner')
# # print(merge_csv.head())

# #Data Transformation

# if set(['like', 'comment', 'share']).issubset(df.columns):
#     df['engagemenmt'] = df['like'] + df['share'] + df['comment']


# #Data Modeling
# #Linear Regression to predict like based on share and comment
# from sklearn.linear_model import LinearRegression
# if set(['like', 'share', 'comment']).issubset(df.columns):
#     x = df['share', 'comment']
#     y = df['like']

#     model=LinearRegression()
#     model.fit(x,y)

#     print("Model coefficients:", model.coef_)
#     print("Model intercept:", model.intercept_)


# #Transposing data

# print(df.head().T)


# #Sort Data 
# if 'like' in df.columns:
#     sorted_df = df.sort_values(by = 'like', ascending = false)
#     print(sorted_df.head())


# #Data Subset

# if 'like' in df.columns:
#     popular_posts = df[df('like') > 1000]
#     print(popular_posts.head())



# #Data Visualization
# print("DATA VISUALIZATION ---------------------------------------------------->")

# sns.set_theme(style="whitegrid")
# print(df.select_dtypes(include=['int64', 'float64']).columns)   

# numeric_cols = ['like', 'comment', 'share']

# # print("Box Plot")
# # plt.figure(figsize=(8,6))
# # sns.boxplot(data= df[numeric_cols])
# # plt.title("Box Plot of Numeric Columns")
# # plt.show()


# # print("Histogram")
# plt.figure(figsize=(8,6))
# plt.hist(df['Open'], bins=30, color='skyblue', edgecolor='black')
# plt.title("Histogram of Likes")
# plt.xlabel('Number of likes')
# plt.ylabel('Frequency')
# plt.show()

# # ------- 3️⃣ SINGLE LINE GRAPH -------
# # plt.figure(figsize=(8, 6))
# # plt.plot(df['Open'], color='purple')
# # plt.title('Single Line Graph of Likes Over Index')
# # plt.xlabel('Index')
# # plt.ylabel('Likes')
# # plt.show()


# # ------- 4️⃣ MULTIPLE LINE GRAPH -------
# # plt.figure(figsize=(10, 6))
# # plt.plot(df['Open'], label='Likes', color='blue')
# # plt.plot(df['High'], label='Shares', color='green')
# # plt.plot(df['Low'], label='Comments', color='red')
# # plt.title('Multiple Line Graph of Likes, Shares, Comments')
# # plt.xlabel('Index')
# # plt.ylabel('Count')
# # plt.legend()
# # plt.show()




# REVIEW SCRAPER---------------------------------------------->

import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = 'https://books.toscrape.com/catalogue/page-{}.html'

titles = []
prices = []
ratings = []

for page in range(1, 4):  # First 3 pages
    url = base_url.format(page)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    books = soup.find_all('article', class_='product_pod')
    
    for book in books:
        title = book.h3.a['title']
        price = book.find('p', class_='price_color').text
        rating = book.p['class'][1]  # star-rating class
        
        titles.append(title)
        prices.append(price)
        ratings.append(rating)

df = pd.DataFrame({
    'Title': titles,
    'Price': prices,
    'Rating': ratings
})

df.to_csv('books_scraped.csv', index=False)
print('✅ Scraped data saved to books_scraped.csv')


# ✅ What it does:
# Loops over 3 pages of Books to Scrape
# For each page:
# → Downloads the HTML using requests
# → Parses the HTML using BeautifulSoup
# → Finds each book’s title, price, and rating
# Stores everything into a pandas DataFrame
# Saves the data as a CSV file
# We are scraping from the website →
# 🌐 https://books.toscrape.com