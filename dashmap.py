import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


#import corona virus dataset

df = pd.read_csv('Datasets/Data.csv')
df.drop(['Province/State','Lat', 'Long'],axis=1,inplace=True)

## Handling Different Names
df = df.replace(to_replace ="US", 
                 value ="United States of America")

df = df.replace(to_replace ="Korea, South", 
                 value ="South Korea")

df = df.replace(to_replace ="Korea, North", 
                 value ="North Korea")

df = df.replace(to_replace ="South Sudan", 
                 value ="S. Sudan")
df = df.replace(to_replace ="Central African Republic", 
                 value ="Central African Rep.")
df = df.replace(to_replace ="Western Sahara", 
                 value ="W. Sahara")
df = df.replace(to_replace ="Bosnia and Herzegovina", 
                 value ="Bosnia and Herz.")
df = df.replace(to_replace ="Equatorial Guinea", 
                 value ="Eq. Guinea")
df = df.replace(to_replace ="Taiwan*", 
                 value ="Taiwan")
df = df.replace(to_replace ="Dominican Republic", 
                 value ="Dominican Rep.")
df = df.replace(to_replace ="Eswatini", 
                 value ="eSwatini")
df = df.replace(to_replace ="North Macedonia", 
                 value ="Macedonia")



#print(df[df['Country/Region']=='US'])

# Handling countries with multiple values
countries = ['China','France','Canada','Australia','Netherlands','Denmark','United Kingdom'] #

for c in countries:
    df_temp = df[df['Country/Region']==c]
    count_list = list(df_temp.sum())
    count_list[0] = c
    new_count_list = []
    for i in count_list:
        new_count_list.append([i])

    #print((count_list))
    column_list = list(df.columns)
    #print(column_list)

    #Deleting existing rows
    delete_row = df[df["Country/Region"]==c].index
    df = df.drop(delete_row)
    dictionary = {column_list[i]: new_count_list[i] for i in range(len(new_count_list))}
    df_new = pd.DataFrame.from_dict(dictionary)
    df = df.append(df_new)

cols = list(df.columns)
cols.pop(0)


## Melting the data
df = pd.melt(df, id_vars =['Country/Region'], value_vars = cols)
#print(df)


data_slider = []


my_colorsc=[[0, 'lightcyan'],
            [0.001, 'darkgreen'],
            [0.005, 'green'],
            [0.05, 'lime'],     
            [0.1, 'lime'],#orange
            [0.2, 'lawngreen'],
            [0.3, 'yellow'],
            [0.4, 'yellow'],
            [0.5, 'gold'], 
            [0.6, 'orange'],
            [0.7, 'darkorange'],
            [0.8, 'orangered'],
            [0.9, 'red'],
            [1, 'darkred']]

for date in df['variable'].unique():

    # I select the date
    df_date = df[df['variable'] == date]

    for col in df_date.columns:  #transform the columns into string type:
        df_date[col] = df_date[col].astype(str)

    ### create the dictionary with the data for the current date
    data_one_date = dict(
                        type='choropleth',
                        locations = df_date['Country/Region'],
                        z=df_date['value'].astype(float),
                        locationmode='country names',
                        colorscale = my_colorsc
                        )

    data_slider.append(data_one_date)




## steps for the slider
steps = []

for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label=cols[i]) # label to be displayed for each step (date)'Year {}'.format(i + 1960)
    step['args'][1][i] = True
    steps.append(step)


sliders = [dict(active=0, pad={"t": 1}, steps=steps)] 


# layout (including slider option)
layout = dict(geo=dict(scope='world', showcountries = True, projection={'type': 'equirectangular'}), sliders=sliders)


plotly.offline.plot({
    "data": data_slider,
    "layout": layout
})






