import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import time


def main():
    filepath_map = '../data/skorea-provinces-2018-geo.json'
    filepath_corona = '../data/data_handmade_0306to0719.csv'

    df_map, df_corona = data_loader(filepath_map, filepath_corona)
    list_date = df_corona['date']
    df_corona = df_corona.drop(columns=['date'])
    values = df_corona.values.reshape(1, -1)[0]
    vmin, vmax = min(values), max(values)
    print(vmin, vmax)
    df_corona['date'] = list_date

    for idx, date_ in enumerate(sorted(list_date)):
        save_as_image(df_map, df_corona, vmin, vmax, idx, date_=date_)
        print(date_)

        
def convert_all_to_int(l):
    values = []
    for v in l:
        if type(v) is str:
            values.append(float(v.replace(',', '')))
        else:
            values.append(float(v))
    
    return values

    
def data_loader(filepath_map,  filepath_corona):
    df_map = gpd.read_file(filepath_map, encoding='utf8')
    df_corona = pd.read_csv(filepath_corona)
    for province in df_corona.columns[1:]:
        df_corona[province] = convert_all_to_int(df_corona[province])
    
    return df_map, df_corona


def save_as_image(df_map, df_corona, vmin, vmax, idx, date_=None):
    df_temp = df_corona[df_corona['date']==date_].drop(columns=['date'])
    df_temp = df_temp.transpose()
    df_temp = df_temp.reset_index().rename(index=str, columns={'index': 'province', idx: 'new_confirmed'})
    merged = df_map.set_index('name_eng').join(df_temp.set_index('province'))
    
    fig, ax = plt.subplots(1, figsize=(11, 11))
    ax.axis('off')
    merged.plot(column='new_confirmed', cmap='gray_r', linewidth=0.8, ax=ax, edgecolor='0.8', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.savefig('../result/handmade/map_{}.png'.format(date_), dpi=30)
    time.sleep(1)
    plt.close()
    plt.clf()


if __name__=="__main__":
    main()
    