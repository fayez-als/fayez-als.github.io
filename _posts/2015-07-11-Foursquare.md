---
layout:     post
title:      Analyzing neighborhoods in Jeddah City.
date:       2020-03-14
summary:    retrieving and cleaning venues data through Foursquare API.
categories: jekyll pixyll
---

Often it is difficult to choose neighborhoods when moving to a new city. Given certain criteria, can more informed decisions be made? in this quick project I'll demonstrate how to retrieve, clean, aggregate, and analyze differend neighboroods' venues in Jeddah (or any other cities) using Foursquare API.




```python
CLIENT_ID = '' # your Foursquare ID
CLIENT_SECRET = ''# your Foursquare Secret
VERSION = '20180604'
LIMIT = 30
```
    




```python
address = 'Jeddah'
geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
categoryID = '4bf58dd8d48988d110941735'
radius = 200000
url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&categoryId={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, categoryID, radius, LIMIT)
results = requests.get(url).json()
venues = results['response']['venues']
dataframe = json_normalize(venues)
```


```python
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']




def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]
```


```python
neighborhoods = pd.read_csv('Jed.csv',names=['Neighborhood','Latitude','Longitude'])
neighborhoods['Neighborhood']=neighborhoods['Neighborhood'].str.capitalize()
```
```python
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]
```


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
            
        
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        results = requests.get(url).json()["response"]['groups'][0]['items']
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
jeddah_venues = getNearbyVenues(names=neighborhoods['Neighborhood'],
                                   latitudes=neighborhoods['Latitude'],
                                   longitudes=neighborhoods['Longitude']
                                  )
```


```python

```


```python
jeddah_onehot = pd.get_dummies(jeddah_venues[['Venue Category']], prefix="", prefix_sep="")
jeddah_onehot['Neighborhood'] = jeddah_venues['Neighborhood'] 
fixed_columns = [jeddah_onehot.columns[-1]] + list(jeddah_onehot.columns[:-1])
jeddah_onehot = jeddah_onehot[fixed_columns]
jeddah_grouped = jeddah_onehot.groupby('Neighborhood').sum().reset_index()
num_top_venues = 5
for hood in jeddah_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = jeddah_grouped[jeddah_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```



And here are the top cities in jeddah by Venues
    ----Albsateen----
                           venue  freq
    0        American Restaurant   6.0
    1                Pizza Place   4.0
    2             Ice Cream Shop   3.0
    3  Middle Eastern Restaurant   3.0
    4                Coffee Shop   3.0
    
    
    ----Alkhaldeya----
                venue  freq
    0     Coffee Shop   5.0
    1  Breakfast Spot   3.0
    2    Burger Joint   3.0
    3          Bakery   2.0
    4  Sandwich Place   2.0
    
    
    ----Almurjan----
                    venue  freq
    0  Seafood Restaurant   5.0
    1         Coffee Shop   4.0
    2          Restaurant   4.0
    3       Grocery Store   3.0
    4         Pizza Place   2.0
    
    
    ----Alnahdha----
                   venue  freq
    0        Coffee Shop   6.0
    1             Bakery   4.0
    2  Indian Restaurant   4.0
    3        Pizza Place   3.0
    4       Dessert Shop   3.0
    
    
    ----Alrawdha----
                    venue  freq
    0        Dessert Shop   5.0
    1  Chinese Restaurant   5.0
    2         Coffee Shop   4.0
    3         Pizza Place   2.0
    4         Tailor Shop   1.0
        


```python

```

