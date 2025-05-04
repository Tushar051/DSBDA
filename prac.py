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




# HIVEQL----------------------------------------------->

# CREATE DATABASE IF NOT EXISTS employee_db;
# USE employee_db;

# CREATE TABLE IF NOT EXISTS emp_info (
#     EmpID INT,
#     EmpName STRING,
#     EmpDesignation STRING,
#     Dept STRING
# )
# ROW FORMAT DELIMITED
# FIELDS TERMINATED BY ','
# STORED AS TEXTFILE;

# CREATE TABLE IF NOT EXISTS emp_salary (
#     EmpID INT,
#     Salary FLOAT,
#     Leaves_Taken INT
# )
# ROW FORMAT DELIMITED
# FIELDS TERMINATED BY ','
# STORED AS TEXTFILE;

# ALTER TABLE emp_info ADD COLUMNS (Age INT);

# DROP TABLE emp_info;
# DROP TABLE emp_salary;


# CREATE EXTERNAL TABLE IF NOT EXISTS emp_external (
#     EmpID INT,
#     EmpName STRING,
#     EmpDesignation STRING,
#     Dept STRING
# )
# ROW FORMAT DELIMITED
# FIELDS TERMINATED BY ','
# STORED AS TEXTFILE
# LOCATION '/user/hive/warehouse/emp_external';

# LOAD DATA LOCAL INPATH '/path/to/emp_info.csv' INTO TABLE emp_info;
# LOAD DATA LOCAL INPATH '/path/to/emp_salary.csv' INTO TABLE emp_salary;

# INSERT INTO TABLE emp_info VALUES (101, 'Alice', 'Manager', 'HR', 35);
# INSERT INTO TABLE emp_salary VALUES (101, 75000, 2);

# SELECT i.EmpID, i.EmpName, i.EmpDesignation, i.Dept, s.Salary, s.Leaves_Taken
# FROM emp_info i
# JOIN emp_salary s
# ON i.EmpID = s.EmpID;




# HBASE and  HIVEQL Integration----------------------------------------------->

# Note: Use the below commands before starting the hbase.
# 1. sudo service zookeeper-server start
# 2. sudo service hadoop-hdfs-namenode start
# 3. sudo service hadoop-hdfs-datanode start
# 4. sudo service hbase-master start
# 5. sudo service hbase-regionserver start
# 6. hbase shell

# Alternative for step 2 and 3 is:
# start-all.sh

# > to stop all hdfs services at the end use
# stop-all.sh


# hbase shell

# create 'employee_hbase', 'info'
# put 'employee_hbase', '101', 'info:name', 'Alice'
# put 'employee_hbase', '101', 'info:designation', 'Manager'
# put 'employee_hbase', '101', 'info:dept', 'HR'
# put 'employee_hbase', '101', 'info:salary', '75000'



# run this in hive
# CREATE EXTERNAL TABLE employee_hive (
#     EmpID STRING,
#     name STRING,
#     designation STRING,
#     dept STRING,
#     salary STRING
# )
# STORED BY 'org.apache.hadoop.hive.hbase.HBaseStorageHandler'
# WITH SERDEPROPERTIES (
#     "hbase.columns.mapping" = ":key,info:name,info:designation,info:dept,info:salary"
# )
# TBLPROPERTIES (
#     "hbase.table.name" = "employee_hbase"
# );


# SELECT * FROM employee_hive;

# INSERT INTO TABLE employee_hive VALUES ('102', 'Bob', 'Developer', 'IT', '60000');


# Now if you go back to hbase shell:
# get 'employee_hbase', '102'




# HBASE Creating, Dropping, and altering Database tablesUsing Hbase------------------------------------------------->   

#Create Table:
# hbase(main):002:0&gt; create 'flight','finfo','fsch'

# #Table Created-list
# hbase(main):003:0&gt; list

# #Insert records in created table
# hbase(main):004:0> put 'flight',1,finfo:source,'pune'
# hbase(main):008:0&gt; put &#39;flight&#39;,1,&#39;finfo:dest&#39;,&#39;mumbai&#39;
# hbase(main):010:0&gt; put &#39;flight&#39;,1,&#39;fsch:at&#39;,&#39;10.25a.m.&#39;
# hbase(main):011:0&gt; put &#39;flight&#39;,1,&#39;fsch:dt&#39;,&#39;11.25 a.m.&#39;
# hbase(main):012:0&gt; put &#39;flight&#39;,1,&#39;fsch:delay&#39;,&#39;5&#39;
# hbase(main):015:0&gt; put &#39;flight&#39;,2,&#39;finfo:source&#39;,&#39;pune&#39;
# hbase(main):016:0&gt; put &#39;flight&#39;,2,&#39;finfo:dest&#39;,&#39;kolkata&#39;
# hbase(main):017:0&gt; put &#39;flight&#39;,2,&#39;fsch:at&#39;,&#39;7.00a.m.&#39;
# hbase(main):018:0&gt; put &#39;flight&#39;,2,&#39;fsch:dt&#39;,&#39;7.30a.m.&#39;
# hbase(main):019:0&gt; put &#39;flight&#39;,2,&#39;fsch:delay&#39;,&#39;2&#39;

# #Display Records from Table ‘flight’
# hbase(main):031:0&gt; scan 'flight'

# #Alter Table (add one more column family)
# hbase(main):036:0&gt; alter 'flight',NAME=>'revenue'

# #Insert records into added column family
# hbase(main):038:0&gt; put 'flight',4,'revenue:rs' , '45000'

# #Check the updates
# hbase(main):039:0&gt; scan 'flight'

# #Delete Column family
# hbase(main):040:0&gt; alter flight,NAME=>'revenue',METHOD=>'delete'
# #changes Reflected in Table
# hbase(main):041:0&gt; scan flight

# #Drop Table
# #Create Table for dropping
# hbase(main):046:0* create 'tb1' , 'cf'

# #Drop Table
# hbase(main):048:0&gt; drop 'tb1'

# #Disable table
# hbase(main):049:0&gt; disable 'tb1'
# hbase(main):050:0&gt; drop 'tb1'
# hbase(main):051:0&gt; list

# #Read data from table for row key 1:
# hbase(main):052:0&gt; get 'flight1'

# #Read data for particular column from HBase table:
# hbase(main):053:0&gt; get 'flight1', '1', COLUM>'finfo:source'