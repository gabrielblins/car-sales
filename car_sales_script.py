'''
Exploring eBay Car Sales Data
This is the first project of [DCA0305 - Machine Learning Based Systems Design]
(https://github.com/ivanovitchm/mlops#dca0305---machine-learning-based-systems-design)
class and the guided project of "Practice data cleaning and data exploration
using pandas" module of Dataquest(https://dataquest.io).In this project I will
work with the eBay Kleinanzeigen used cars dataset, Im using a modified version
of this dataset provided by Dataquest. You can find the unmodified dataset in
this link(https://data.world/data-society/used-cars-data).
The aim of this project is to clean and analyze the data of the used cars,
trying to extract some useful information and applying
the concepts learned during the course.
Let's get started!
By: Gabriel Lins (https://github.com/gabrielblins)
'''
# Importing the libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Defining functions
def plot_bar_chart(data_frame,column_name, is_str=True):
    '''
    This function plots a bar chart of the dataframe and save it
    '''
    figure, axis = plt.subplots(figsize=(10, 8))
    if is_str:
        (data_frame[column_name].str[:10]
                                .value_counts(normalize=True, dropna=False)
                                .sort_index()
                                .plot(kind='bar', ax=axis))
    else:
        (data_frame[column_name].value_counts(normalize=True, dropna=False)
                                .sort_index().plot(kind='bar', ax=axis))
    figure.savefig(f'plots/{column_name}.png')

if __name__ == '__main__':
    # Reading the csv file using pandas
    autos = pd.read_csv('autos.csv', encoding='latin-1')
    # Changing column names from camelCase to snake_case
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    autos.rename({'yearOfRegistration': 'registration_year',
                  'monthOfRegistration': 'registration_month',
                  'notRepairedDamage': 'unrepaired_damage',
                  'dateCreated': 'ad_created',
                  'powerPS': 'power_ps'}, axis=1, inplace=True)
    autos.columns = [pattern.sub('_', col).lower() for col in autos.columns]
    # Dropping the seller, offer_type, and nr_of_pictures columns
    # because they don't bring any useful information of our data
    autos = autos.drop(['seller', 'offer_type', 'nr_of_pictures'], axis=1)
    # Changing type of numerical columns that are stored as text
    autos.price = (autos.price
                        .str.replace('$', '')
                        .str.replace(',', '')
                        .astype(np.float64))
    autos.odometer = (autos.odometer
                      .str.replace('km', '')
                      .str.replace(',', '')
                      .astype(np.float64))
    autos = autos.rename({'odometer': 'odometer_km'}, axis=1)
    # Removing Outliers
    # Filtering the dataframe to only keep the cars that have
    # a price greater than 0 and less than 120000
    autos = (autos[autos.price.between(autos.price.quantile(0.03),
                                       autos.price.quantile(0.999))])
    # Looking at date columns
    # Saving the plot of the bar chart of date_crawled column
    plot_bar_chart(autos, 'date_crawled')
    # Saving the plot of the bar chart of ad_created column
    plot_bar_chart(autos, 'ad_created')
    # Saving the plot of the bar chart of last_seen column
    plot_bar_chart(autos, 'last_seen')
    # Filtering the dataframe by the registration_year column,
    # fitting between the years 1900 and 2016
    autos = autos[autos.registration_year.between(1900, 2016)]
    # Saving the plot of the bar chart of registration_year column
    plot_bar_chart(autos, 'registration_year', is_str=False)
    # Creating a Series with the Top 20 brands most common in the dataset
    top20_brands = autos.brand.value_counts().head(20)
    # Creating a Dictionary with the Top 20 brands and their respective mean
    # price
    mean_price_top20_brands = {}
    for brand in top20_brands.index:
        mean_price_top20_brands[brand] = autos[autos.brand ==
                                               brand].price.mean()
    # Creating a Dictionary with the Top 20 brands and their respective mean km
    mean_km_top20_brands = {}
    for brand in top20_brands.index:
        mean_km_top20_brands[brand] = (autos[autos.brand == brand].odometer_km
                                       .mean())
    # Creating a DataFrame with the mean price and mean km for the top 20
    # brands
    best_mean_prices = pd.Series(mean_price_top20_brands)
    df_top_brands = pd.DataFrame(best_mean_prices, columns=['mean_price'])
    df_top_brands['mean_km'] = pd.Series(mean_km_top20_brands)
    # Creating a plot with the correlation between
    # the mean price and mean km for the top 20 brands
    corr = df_top_brands.corr()['mean_price'][1]
    fig_brand, ax_top_brands = plt.subplots(figsize=(10, 8))
    df_top_brands.plot(
        x='mean_price',
        y='mean_km',
        kind='scatter',
        title='Mean price vs. mean km (ρ = {:.2f})'.format(corr),
        ax=ax_top_brands)
    fig_brand.savefig('plots/mean_price_vs_mean_km.png')
