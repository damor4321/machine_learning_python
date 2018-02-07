
# coding: utf-8

# # Fire up graphlab create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[4]:

import graphlab


# In[5]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Load some house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.

# In[6]:

sales = graphlab.SFrame('datasets/kc_house_data.gl/')


# In[7]:

sales


# # Exploring the data for housing sales 

# The house price is correlated with the number of square feet of living space.

# In[8]:

graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")


# # Create a simple regression model of sqft_living to price

# Split data into training and testing.  
# We use seed=0 so that everyone running this notebook gets the same results.  In practice, you may set a random seed (or let GraphLab Create pick a random seed for you).  

# In[9]:

train_data,test_data = sales.random_split(.8,seed=0)


# ## Build the regression model using only sqft_living as a feature

# In[10]:

sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)


# # Evaluate the simple model

# In[11]:

print test_data['price'].mean()


# In[12]:

print sqft_model.evaluate(test_data)


# RMSE of about \$255,170!

# # Let's show what our predictions look like

# Matplotlib is a Python plotting library that is also useful for plotting.  You can install it with:
# 
# 'pip install matplotlib'

# In[13]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[14]:

plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')


# Above:  blue dots are original data, green line is the prediction from the simple regression.
# 
# Below: we can view the learned regression coefficients. 

# In[15]:

sqft_model.get('coefficients')


# # Explore other features in the data
# 
# To build a more elaborate model, we will explore using more features.

# In[16]:

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']


# In[17]:

sales[my_features].show()


# In[18]:

sales.show(view='BoxWhisker Plot', x='zipcode', y='price')


# Pull the bar at the bottom to view more of the data.  
# 
# 98039 is the most expensive zip code.

# # Build a regression model with more features

# In[19]:

my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)


# In[20]:

print my_features


# ## Comparing the results of the simple model with adding more features

# In[21]:

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)


# The RMSE goes down from \$255,170 to \$179,508 with more features.

# # Apply learned models to predict prices of 3 houses

# The first house we will use is considered an "average" house in Seattle. 

# In[22]:

house1 = sales[sales['id']=='5309101200']


# In[23]:

house1


# <img src="http://info.kingcounty.gov/Assessor/eRealProperty/MediaHandler.aspx?Media=2916871">

# In[24]:

print house1['price']


# In[25]:

print sqft_model.predict(house1)


# In[26]:

print my_features_model.predict(house1)


# In this case, the model with more features provides a worse prediction than the simpler model with only 1 feature.  However, on average, the model with more features is better.

# ## Prediction for a second, fancier house
# 
# We will now examine the predictions for a fancier house.

# In[27]:

house2 = sales[sales['id']=='1925069082']


# In[28]:

house2


# <img src="https://ssl.cdn-redfin.com/photo/1/bigphoto/302/734302_0.jpg">

# In[29]:

print sqft_model.predict(house2)


# In[30]:

print my_features_model.predict(house2)


# In this case, the model with more features provides a better prediction.  This behavior is expected here, because this house is more differentiated by features that go beyond its square feet of living space, especially the fact that it's a waterfront house. 

# ## Last house, super fancy
# 
# Our last house is a very large one owned by a famous Seattleite.

# In[31]:

bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}


# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Bill_gates%27_house.jpg/2560px-Bill_gates%27_house.jpg">

# In[32]:

print my_features_model.predict(graphlab.SFrame(bill_gates))


# The model predicts a price of over $13M for this house! But we expect the house to cost much more.  (There are very few samples in the dataset of houses that are this fancy, so we don't expect the model to capture a perfect prediction here.)

# ## 1. Compute the average price from the neighbourhood with the highest average house sale price (zipcode 98039)

# In[33]:

average_price_for_98039_zipcode = sales[sales['zipcode'] == "98039"]['price'].mean()


# In[34]:

average_price_for_98039_zipcode


# ## 2. Select the houses that have ‘sqft_living’ higher than 2000 sqft but no larger than 4000 sqft. Then compute the fraction of the all houses have ‘sqft_living’ in this range.

# In[35]:

sales_from_2000_to_4000_sqft = sales[(sales['sqft_living']> 2000) & (sales['sqft_living'] <= 4000)]


# In[36]:

sales_from_2000_to_4000_sqft


# In[37]:

sales_from_2000_to_4000_sqft.num_rows()


# In[38]:

(sales_from_2000_to_4000_sqft.num_rows() * 100)/ sales.num_rows()


# 9118 houses in the range from 2000 to 4000 sq.ft. This is 42% of the all houses

# ## 3. Building a regression model with several more features

# In[39]:

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[40]:

advanced_features


# In[41]:

advanced_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)


# In[42]:

print my_features_model.evaluate(test_data)
print advanced_features_model.evaluate(test_data)


# In[43]:

rmse_diff = my_features_model.evaluate(test_data)['rmse'] - advanced_features_model.evaluate(test_data)['rmse']


# In[44]:

rmse_diff


# In[45]:

train_data2,test_data2 = sales.random_split(.8,seed=0)


# In[46]:

train_data.num_rows() == train_data2.num_rows()


# In[47]:

train_data2['price'].mean()
train_data['price'].mean()


# In[48]:

train_data['price'].mean()


# In[ ]:



