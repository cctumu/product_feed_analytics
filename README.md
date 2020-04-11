# product_feed_analytics
This project is validating product categories, titles and product types. It tests the matching rate of  key words in product titles, product types and google categories.

### Why it matters?
This project is used in a recommendation system. We get all the data from customer's website. If there are some mistakes in product grouping in google category, the recommendation would not be accurate related to the wrong categories. So this analysis is using the product name and type to check if the product matches google category.


### What is the main method?

#### Step 1: 
Filtering nouns from product titles and types which could be the key words for each item. 
#### Step 2:
Using special module in python to compare sequences of key words with google category in a fuzzy range. 
#### Step 3: 
Sorting data by the matching rates, then we could easily find the wrong categories.

