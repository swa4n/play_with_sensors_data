from datetime import datetime
import re
from math import sin, cos, sqrt, atan2, radians

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


DATA_PATH = "./data/"
THRESH = 100  # maxumum distance in meters that determines if 2 locations are close
N = 2  # maximum number of favourite locations to be considered, favourite refers to locations where user spends most time in


class User:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.get_data()

    def get_data(self):
        data = pd.read_csv(DATA_PATH+self.filename, sep=';')
        data.columns = ['latitude', 'longitude', 'start_date', 'duration']
        return data

    def has_visted(self, location):
        ''' returns a list of visited locations that are close within defined threshold to the location in query. 
            the methods loops over the dataframe of size n*m once. it makes O(n) in time and O(n*m) in space.            
        '''
        res = []
        for index, row in self.data.iterrows():
            loc = (row['latitude'], row['longitude'])
            if is_close(loc, location):
                res.append(loc)
        if res == []:
            print("User has never visited this location")
            return None
        return res

    def get_home_work_locations(self):
        '''We want to first cluster locations depending on when during the day, the user has visited these places. Then, for every cluster, we calculate the overall duration the user spends
        in every location. The locations with the maximum overall durations are the most potential to be the ones we are looking for.

        We continue afterwards, filtering the results by calculating the closeness of the top 3 found locations (closeness defined by THRESHOLD). If 2 locations are close, then they refer 
        to the same visited place and therfore durations of both of them will add up. If not, then it is more likely that the user has more than one home/work address.        
        '''
        df = self.data
        df = cluster_locations(df)
        df.set_index('start_date', inplace=True)

        # a person is more likely to be home at night
        home_df = df.between_time('22:00', '06:00')
        # a person is more likely to spend 9AM to 6PM at work
        work_df = df.between_time('09:00', '18:00')
        # weekends locations will help us define home locations
        weekends = df[df['weekday'].between(5, 6)]
        df.reset_index(inplace=True)

        favourate_home_labeled_locations = favourite_locations(home_df, N)
        favourate_work_labeled_locations = favourite_locations(work_df, N)
        favourate_weekend_labeled_locations = favourite_locations(weekends, N)

        home_locations = []
        work_locations = []
        for loc in favourate_home_labeled_locations:
            home_locations.append(get_coordinates(df, loc))
        for loc in favourate_weekend_labeled_locations:
            home_locations.append(get_coordinates(df, loc))
        for loc in favourate_work_labeled_locations:
            work_locations.append(get_coordinates(df, loc))

        # check closeness: if two home locations are less than THRESHOLD (100 meters) they are more likely to refer to the same place
        home = unique_locations(home_locations)
        work = unique_locations(work_locations)
        return {'Home': home, 'Work': work}

# helper functions


def is_close(location1, location2, threshold=THRESH):
    '''returns true if the locations are close, false otherwise. closeness is defined by a threshold in meters. 
       source: https://bit.ly/314D75A
    '''
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(location1[0])
    lon1 = radians(location1[1])
    lat2 = radians(location2[0])
    lon2 = radians(location2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c * 1000  # distance calculated in meters
    if distance <= threshold:
        return True
    else:
        return False


def preprocess(df):
    ''' preprocess data, apply some feature engineering to prepare for the recognition of home and work locations. returns a dataframe of ready-to-feed-model data'''

    # standardize latitude and longitude values for clustering
    scaler = StandardScaler()
    X = df[['latitude', 'longitude']]
    X = scaler.fit_transform(X)

    df['latitude_standardized'] = X[:, 0]
    df['longitude_standardized'] = X[:, 1]

    # set date_time in canonic form and extract days of week
    df['start_date'] = df['start_date'].apply(lambda x: get_time(x))
    weekday = []
    for index, row in df.iterrows():
        weekday.append(row['start_date'].weekday())
    df['weekday'] = weekday     # Monday = 0 and Sunday = 6
    return df


def get_time(time_string):
    time = re.sub(r'[-+]', '', time_string)
    return datetime.strptime(time[:-4], '%Y%m%d%H%M')


def cluster_locations(df):
    df = preprocess(df)
    X = df[['latitude_standardized', 'longitude_standardized']]
    kmeans = KMeans(20)
    kmeans.fit(X)
    identified_locations = kmeans.fit_predict(X)
    df['clustered_locations'] = identified_locations
    return df


def favourite_locations(df, n=N):
    '''returns the top n locations in df where user spends most time'''
    locations = df.clustered_locations.unique()
    total_duration_per_location = {}

    for loc in locations:
        df_loc = df[df['clustered_locations'] == loc]
        total_duration_per_location[loc] = df_loc.duration.sum()

    values = np.array(list(total_duration_per_location.values()))
    top_n_durations = values[np.argsort(values)[-n:]]
    top_n_locations = []

    for location in total_duration_per_location.keys():
        if total_duration_per_location[location] in top_n_durations:
            top_n_locations.append(location)

    return top_n_locations


def get_coordinates(df, label):
    i = df[['latitude', 'longitude', 'clustered_locations']].query(
        'clustered_locations=={}'.format(label)).sample(1, random_state=0).index
    loc_ = [df.iloc[i]['latitude'].iloc[0],
            df.iloc[i]['longitude'].iloc[0]]
    return loc_


def unique_locations(locations_list):
    res = [locations_list[0]]
    i = 1
    while i < len(locations_list):
        tst = False
        for loc in res:
            if is_close(loc, locations_list[i]):
                tst = True
                break
        if tst == False:
            res.append(locations_list[i])
        i += 1
    return res
