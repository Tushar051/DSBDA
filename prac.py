import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('FB_data.csv')
print(df.head())


# Data Cleaning

print(df.isnull().sum())
df = df.fillna(0)
df = df.drop_duplicates()
print(df.shape)

#Data Integration

# df2 = pd.read_csv('')
# merge_csv = pd.merge(df, df2, on='post_id', how='inner')
# print(merge_csv.head())

#Data Transformation

if set(['like', 'comment', 'share']).issubset(df.columns):
    df['engagemenmt'] = df['like'] + df['share'] + df['comment']


#Data Modeling
#Linear Regression to predict like based on share and comment
from sklearn.linear_model import LinearRegression
if set(['like', 'share', 'comment']).issubset(df.columns):
    x = df['share', 'comment']
    y = df['like']

    model=LinearRegression()
    model.fit(x,y)

    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)


#Transposing data

print(df.head().T)


#Sort Data

if 'like' in df.columns:
    sorted_df = df.sort_values(by = 'like', ascending = false)
    print(sorted_df.head())


#Data Subset

if 'like' in df.columns:
    popular_posts = df[df('like') > 1000]
    print(popular_posts.head())



#Data Visualization
print("DATA VISUALIZATION ---------------------------------------------------->")

sns.set_theme(style="whitegrid")
print(df.select_dtypes(include=['int64', 'float64']).columns)   

numeric_cols = ['like', 'comment', 'share']

# print("Box Plot")
# plt.figure(figsize=(8,6))
# sns.boxplot(data= df[numeric_cols])
# plt.title("Box Plot of Numeric Columns")
# plt.show()


# print("Histogram")
plt.figure(figsize=(8,6))
plt.hist(df['Open'], bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Likes")
plt.xlabel('Number of likes')
plt.ylabel('Frequency')
plt.show()

# ------- 3️⃣ SINGLE LINE GRAPH -------
# plt.figure(figsize=(8, 6))
# plt.plot(df['Open'], color='purple')
# plt.title('Single Line Graph of Likes Over Index')
# plt.xlabel('Index')
# plt.ylabel('Likes')
# plt.show()


# ------- 4️⃣ MULTIPLE LINE GRAPH -------
# plt.figure(figsize=(10, 6))
# plt.plot(df['Open'], label='Likes', color='blue')
# plt.plot(df['High'], label='Shares', color='green')
# plt.plot(df['Low'], label='Comments', color='red')
# plt.title('Multiple Line Graph of Likes, Shares, Comments')
# plt.xlabel('Index')
# plt.ylabel('Count')
# plt.legend()
# plt.show()


