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



#Data Modeling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv('facebook_data.csv')


X = df.drop('Type', axis=1)   # Replace 'Type' with your target column
y = df['Type']

    # Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

    # Make predictions
y_pred = model.predict(X_test)

    # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#Data Visualization-------------------------------------------------------------------------------------------------->


sns.set_theme(style="whitegrid")
print(df.select_dtypes(include=['int64', 'float64']).columns)   

numeric_cols = ['like', 'comment', 'share']

#------- Box plot GRAPH -------
plt.figure(figsize=(8,6))
sns.boxplot(data= df[numeric_cols])
plt.title("Box Plot of Numeric Columns")
plt.show()


#-------  Histogram GRAPH -------
plt.figure(figsize=(8,6))
plt.hist(df['Open'], bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Likes")
plt.xlabel('Number of likes')
plt.ylabel('Frequency')
plt.show()

#------- 3️⃣ SINGLE LINE GRAPH -------
plt.figure(figsize=(8, 6))
plt.plot(df['Open'], color='purple')
plt.title('Single Line Graph of Likes Over Index')
plt.xlabel('Index')
plt.ylabel('Likes')
plt.show()


#------- 4️⃣ MULTIPLE LINE GRAPH -------
plt.figure(figsize=(10, 6))
plt.plot(df['Open'], label='Likes', color='blue')
plt.plot(df['High'], label='Shares', color='green')
plt.plot(df['Low'], label='Comments', color='red')
plt.title('Multiple Line Graph of Likes, Shares, Comments')
plt.xlabel('Index')
plt.ylabel('Count')
plt.legend()
plt.show()




# REVIEW SCRAPER----------------------------------------------------------------------------------------------------->

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





# HBASE--------------------------------------------------------------------------------------------------------------->

1. sudo service zookeeper-server start
2. sudo service hadoop-hdfs-namenode start
3. sudo service hadoop-hdfs-datanode start
4. sudo service hbase-master start
5. sudo service hbase-regionserver start
6. hbase shell

Alternative for step 2 and 3 is:
start-all.sh

> to stop all hdfs services at the end use
stop-all.sh



create 'student', 'personal', 'academic'

list

put 'student', '1', 'personal:name', 'Alice'
put 'student', '1', 'personal:age', '22'
put 'student', '1', 'academic:dept', 'CS'
put 'student', '1', 'academic:grade', 'A'

get 'student', '1'  #Retrieve a specific row

scan 'student'

put 'student', '1', 'academic:grade', 'A+' # Update a value

delete 'student', '1', 'personal:age' # Delete a specific column in a row

deleteall 'student', '1' # Delete all columns in a row





# HIVEQL--------------------------------------------------------------------------------------------------------------->
press hive in terminal and run the following code

CREATE DATABASE flight_info;
USE flight_info;

CREATE TABLE flights (
    FlightID STRING,
    Airline STRING,
    Source STRING,
    Destination STRING,
    DepartureTime STRING,
    ArrivalTime STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;


CREATE TABLE bookings (
    BookingID STRING,
    FlightID STRING,
    PassengerName STRING,
    SeatNumber STRING,
    Price DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;


INSERT INTO TABLE flights VALUES ('FL101', 'IndiGo', 'Delhi', 'Mumbai', '09:00', '11:30', 'OnTime');
INSERT INTO TABLE bookings VALUES ('BK001', 'FL101', 'John Doe', '12A', 4500);

SELECT * FROM flights;
SELECT * FROM bookings;


    #Alter table---->
ALTER TABLE flights ADD COLUMNS (Status STRING);
INSERT INTO TABLE flights VALUES ('FL123', 'AirIndia', 'Delhi', 'Mumbai', '08:00', '10:00', 'OnTime');


    #Create an external table---->
CREATE EXTERNAL TABLE external_flights (
    FlightID STRING,
    Airline STRING,
    Source STRING,
    Destination STRING,
    DepartureTime STRING,
    ArrivalTime STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LOCATION '/external/flight_data/';

    #Join two tables (get booking + flight details)---->
SELECT b.BookingID, b.PassengerName, f.Airline, f.Source, f.Destination, b.SeatNumber, b.Price
FROM bookings b
JOIN flights f ON b.FlightID = f.FlightID;


    # Load data into tables by csv files--->
LOAD DATA LOCAL INPATH '/path/to/flights.csv' INTO TABLE flights;
LOAD DATA LOCAL INPATH '/path/to/bookings.csv' INTO TABLE bookings;


    
#WordCount cloudera Terminal Commands---------------------------------------------------------------------------------->

step-1
Hadoop fs -put Desktop/Input/Input.txt Input.txt

step-2
hadoop jar WordCount.jar WordCount.WordCount Input.txt dir2

step-3
hadoop fs -cat dir2/part-r-00000


