import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import time


def main():
    filepath_map = '../data/skorea-provinces-2018-geo.json'
    filepath_corona = '../data/data_new_confirmed_200120_200630.csv'
    
    df_map, df_corona = data_loader(filepath_map, filepath_corona)
    vmin, vmax = 0, max(df_corona['new_confirmed'])
    
    for date_ in sorted(set(df_corona['date'])):
        save_as_image(df_map, df_corona, vmin, vmax, date_=date_)
        print(date_)
    

def data_loader(filepath_map,  filepath_corona):
    df_map = gpd.read_file(filepath_map, encoding='utf8')
    df_corona = pd.read_csv(filepath_corona)
    
    df_corona = df_corona[['date', 'province', 'new_confirmed']]
    
    return df_map, df_corona


def save_as_image(df_map, df_corona, vmin, vmax, date_=None):
    df_temp = df_corona[df_corona['date']==date_][['province', 'new_confirmed']]
    merged = df_map.set_index('name_eng').join(df_temp.set_index('province'))
    print(merged)

    fig, ax = plt.subplots(1, figsize=(11, 11))
    ax.axis('off')
    merged.plot(column='new_confirmed', cmap='gray_r', linewidth=0.8, ax=ax, edgecolor='0.8', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.savefig('../result/new_confirmed/map_{}.png'.format(date_), dpi=30)
    time.sleep(1)
    plt.close()
    plt.clf()


if __name__=="__main__":
    main()
