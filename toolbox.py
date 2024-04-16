import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import seaborn as sns

import random
import rasterio
#from rasterio.plot import show
import cartopy.crs as ccrs # probably needs to be installed with pip...
from shapely.geometry import Point

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA


def load_dem(file):
    with rasterio.open(file) as src:
        dem = src.read(1)
        extent = src.bounds

    return dem, extent


def load_weather_station_data(file):
    stations = pd.read_csv(file)
    
    # Create a geometry column with Point objects
    geometry = [Point(x, y) for x, y in zip(stations['xcoord'], stations['ycoord'])]

    # Create a GeoDataFrame from the DataFrame and geometry column
    stations = gpd.GeoDataFrame(stations, geometry=geometry, crs='EPSG:4326')

    # Transform to datetime
    stations.start_date = pd.to_datetime(stations.start_date)
    stations.end_date = pd.to_datetime(stations.end_date)
    
    return stations


def load_weather_data(file):
    w = pd.read_csv(file)

    # change date column to datetime type
    w.date = pd.to_datetime(w.date)
    w.set_index('date', inplace=True)

    # calculate cumulative weather data for moving windows
    #w = calc_cumulative_weather(w)

    return w


def calc_cumulative_weather(df):
    windows = {'2d': 2, '7d': 7,
                '30d': 30, '90d': 90,
                '1y': 365, '3y': 365*3}

    for w in windows:
        df[f'precip_mean_prev_{w}_sum'] = df['precip_mean'].rolling(window=windows[w]).sum()
        df[f'tmean_mean_prev_{w}_mean'] = df['tmean_mean'].rolling(window=windows[w]).mean()

    return df


def load_groundwater_station_data(file):
    stations = pd.read_csv(file)
    
    # correct data types
    stations.start_date = pd.to_datetime(stations.start_date)
    stations.end_date = pd.to_datetime(stations.end_date)
    stations.lifespan = pd.to_timedelta(stations.lifespan)

    # Create a geometry column with Point objects
    geometry = [Point(x, y) for x, y in zip(stations['x'], stations['y'])]

    # Create a GeoDataFrame from the DataFrame and geometry column
    stations = gpd.GeoDataFrame(stations, geometry=geometry, crs='EPSG:32632')
    stations = stations.to_crs('EPSG:4326')
    
    return stations


def load_groundwater_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])

    return df


def calc_aggregate_station_data(stations, data):
    aggregated = data.groupby('station_id').agg({'water_level': 'mean',
                                             'water_depth': 'mean',
                                             'water_temp': 'mean'}).reset_index()

    aggregated = aggregated.rename(columns={'water_level': 'water_level_mean',
                                             'water_depth': 'water_depth_mean',
                                             'water_temp': 'water_temp_mean'})

    stations_agg = pd.merge(stations, aggregated, on='station_id', how='left')

    return stations_agg


def calc_initial_station_data(stations, data, initial_n_years):
    df_list = []

    for id in data.station_id.unique():
        df = data[data.station_id == id]
        start_date = df.date.min()
        end_date = start_date + pd.Timedelta(days=365*initial_n_years)
        df_first_year = df.loc[(df['date'] >= start_date) & (df['date'] < end_date)]
        df_list.append(df_first_year)

    initial_years = pd.concat(df_list)
    initial_years_agg = initial_years.groupby('station_id').agg({'water_depth': ['mean',
                                                                                'std',
                                                                                'min',
                                                                                'max']}
                                                                ).reset_index()

    initial_years_agg.columns = ["_".join(col).rstrip('_') for col in initial_years_agg.columns.values]

    initial_years_agg = initial_years_agg.rename(columns={'water_depth_mean': 'ini_years_water_depth_mean',
                                                        'water_depth_std': 'ini_years_water_depth_std',
                                                        'water_depth_min': 'ini_years_water_depth_min',
                                                        'water_depth_max': 'ini_years_water_depth_max'})

    stations_agg = gpd.GeoDataFrame(pd.merge(stations, initial_years_agg, on='station_id', how='left'),
                                    geometry='geometry')

    return stations_agg


def merge_groundwater_data(data, stations):
    merged = pd.merge(data, stations, how='left')
    merged['water_depth_anomaly'] = merged['water_depth'] - merged['ini_years_water_depth_mean']
    merged.index = merged['date']

    return merged


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true = y_true,
                              y_pred = y_pred)
    
    rmse = mean_squared_error(y_true = y_true,
                              y_pred = y_pred,
                              squared=False)

    mape = mean_absolute_percentage_error(y_true = y_true,
                                          y_pred = y_pred)
    
    r2 = r2_score(y_true = y_true, y_pred = y_pred)

    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
    
    print('MAE', mae)
    print('RMSE', rmse)
    print('MAPE', mape)
    print('R2', r2)
    
    return metrics


def calc_residuals(y_test, y_pred):
    resid = pd.DataFrame()
    resid['observed'] = y_test.copy()
    resid['predicted'] = y_pred.copy()
    resid['residuals'] = resid['predicted'] - resid['observed']
    return resid


def perform_pca(df, n_components=None):
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    # Perform PCA
    if n_components is None:
        n_components = min(df.shape[1], df.shape[0])
    pca = PCA(n_components=n_components).set_output(transform='pandas')
    X_pca = pca.fit_transform(data_scaled)
    
    # Gather PCA statistics
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    
    # Create a DataFrame for explained variance ratio
    df_expl_vari_ratio = pd.DataFrame(explained_variance_ratio,
                                      columns=['Explained Variance Ratio'],
                                      index=[f'PC-{i+1}' for i in range(n_components)])
    
    # Create a DataFrame for components
    df_components = pd.DataFrame(components, 
                                 columns=df.columns, 
                                 index=[f'PC-{i+1}' for i in range(n_components)])
    
    # Concatenate DataFrames
    df_pca_stats = pd.concat([df_expl_vari_ratio, df_components], axis=1)
    
    return pca, X_pca, df_pca_stats.T


def explore_clusters(df, scale=True):
    # DATA SCALING
    if scale == True:
        # Initialise the transformer (optionally, set parameters)
        min_max = MinMaxScaler().set_output(transform="pandas")
        
        # Use the transformer to transform the data
        df = min_max.fit_transform(df)

    wcss = []
    clusters = []
    silhouettes = []

    # CLUSTERING
    for k in range(2, 30):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        n_init=10, max_iter=300)
        
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(df, kmeans.labels_))
        clusters.append(k)

    slope = np.diff(wcss)
    curve = np.diff(slope)
    #elbow = np.argmax(curve) + 1
    #print(clusters[elbow])

    # PLOTTING
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0,0].plot(clusters, wcss)
    axs[0,0].set_xlabel('# of clusters')
    axs[0,0].set_ylabel('inertia')
    
    axs[0,1].plot(clusters, silhouettes)
    axs[0,1].set_xlabel('# of clusters')
    axs[0,1].set_ylabel('silhouette score')
    
    axs[1,0].plot(clusters[:-1], slope)
    axs[1,0].set_xlabel('# of clusters')
    axs[1,0].set_ylabel('inertia slope')
    
    axs[1,1].plot(clusters[:-2], curve)
    axs[1,1].set_xlabel('# of clusters')
    axs[1,1].set_ylabel('inertia curvature')


def apply_clusters(df, n_clusters, scale=True):
    # DATA SCALING
    if scale == True:
        # Initialise the transformer (optionally, set parameters)
        min_max = MinMaxScaler().set_output(transform="pandas")
        
        # Use the transformer to transform the data
        df = min_max.fit_transform(df)
        
     # CLUSTERING
    kmeans = KMeans(n_clusters=n_clusters, #random_state=0,
                    n_init='auto').fit(df)

    return kmeans.labels_, kmeans.cluster_centers_


def plot_clusters(coordinates_df, labels, centers):
    '''
    coordinates_df of the form: df[['x', 'y']]
    '''
    # Plot the points
    plt.scatter(coordinates_df['x'], coordinates_df['y'], c=labels)#, cmap='viridis')
    
    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    
    # Show the plot
    plt.show()
    
    
def tt_split_by_stations(df):
    ids = list(df.station_id.unique())

    # Calculate 80% of the list's length
    num_elements = round(len(ids) * 0.8)
    
    # Randomly select 80% of the elements
    train_ids = random.sample(ids, k=num_elements)

    # create train and test dfs
    train = df.loc[df.station_id.isin(train_ids)]
    test = df.loc[~df.station_id.isin(train_ids)]

    # define X and y
    y_train = train.pop('water_depth')
    y_test = test.pop('water_depth')
    
    reserve_cols_as_info = ['station_id', 'date', 'geometry']
    
    info_train = train[reserve_cols_as_info].copy()
    info_test = test[reserve_cols_as_info].copy()
    
    X_train = train.copy().drop(reserve_cols_as_info, axis=1)
    X_test = test.copy().drop(reserve_cols_as_info, axis=1)

    return train, test, X_train, X_test, y_train, y_test, info_train, info_test


def plot_station_data(df, start_date=None, end_date=None, save=False):
    # Create a figure and an axis
    fig, ax = plt.subplots(4,1, figsize=(10, 10), sharex=True)
    
    sns.lineplot(x="date", y="water_depth",
                 color='tab:orange', alpha=0.5,
                 data=df, ax=ax[0], label='Measured')
    
    sns.lineplot(x="date", y="pred_water_depth",
                 color='tab:red', alpha=0.5,
                 data=df, ax=ax[0], label='Predicted')
    
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Water depth (m)')
    ax[0].legend()
    
    sns.lineplot(x="date", y="residuals",
                 color='k', alpha=0.5,
                 data=df, ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].axhline(y=0, color='k', alpha=0.7)
    ax[1].set_ylabel('Residuals (m)')
    # Create a secondary y-axis
    #ax[1] = ax[0].twinx()
    
    sns.lineplot(x="date", y="tmean_mean_prev_1y_mean", 
                 data=df, ax=ax[2], color='tab:green',
                 #label='Mean Temp. prev. year (°C)'
                 )
    ax[2].set_ylabel('Mean Temp. prev. year (°C)', color='tab:green')
    ax[2].tick_params(axis='y', colors='tab:green')
    
    # Create a secondary y-axis
    ax2_twin = ax[2].twinx()
    
    sns.lineplot(x="date", y="precip_mean_prev_1y_sum", 
                 data=df, ax=ax2_twin, color='tab:blue',
                 #label='Cumulative precip. prev. year (mm)'
                 )
    ax2_twin.set_ylabel('Cumul. precip. prev. year (mm)', color='tab:blue')
    ax2_twin.tick_params(axis='y', colors='tab:blue')
    #ax[2].legend(False)
    
    # Optionally, set labels for the y-axes
    #ax[3].set_ylabel('precip')
    #ax2.set_ylabel('Y2 Label')
    
    # Plot the second DataFrame on the secondary y-axis
    sns.lineplot(x="date", y="tmean_mean_prev_7d_mean", 
                 data=df, ax=ax[3], color='tab:green',
                 #label='Mean temp. prev. year (°C)'
                 )
    ax[3].set_ylabel('Mean Temp. prev. 7 days (°C)', color='tab:green')
    ax[3].tick_params(axis='y', colors='tab:green')
    
    # Create a secondary y-axis
    ax3_twin = ax[3].twinx()
    
    sns.lineplot(x="date", y="precip_mean_prev_7d_sum", 
                 data=df, ax=ax3_twin, color='tab:blue',
                 #label='Cumulative precip. prev. 7 days (mm)'
                 )
    ax3_twin.set_ylabel('Cumul. precip. prev. 7 days (mm)', color='tab:blue')
    ax3_twin.tick_params(axis='y', colors='tab:blue')
    
    #ax[3].legend(False)
    
    # Set the x-axis limits
    if start_date and end_date:
        plt.xlim(start_date, end_date)
    
    # Save the plot
    if save == True:
        plt.savefig('./figs/station_details.png', bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_predicted_data(df, start_date=None, end_date=None, save=False):
    # Create a figure and an axis
    fig, ax = plt.subplots(3,1, figsize=(10, 10), sharex=True)

    sns.lineplot(x="date", y="pred_water_depth",
                 color='tab:red', alpha=0.5,
                 data=df, ax=ax[0], label='Predicted')
    
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Predicted water depth (m)')
    ax[0].legend()
    
    sns.lineplot(x="date", y="tmean_mean_prev_90d_mean", 
                 data=df, ax=ax[1], color='tab:green',
                 #label='Mean Temp. prev. year (°C)'
                 )
    ax[1].set_ylabel('Mean Temp. prev. 90 days (°C)', color='tab:green')
    ax[1].tick_params(axis='y', colors='tab:green')
    
    # Create a secondary y-axis
    ax1_twin = ax[1].twinx()
    
    sns.lineplot(x="date", y="precip_mean_prev_90d_sum", 
                 data=df, ax=ax1_twin, color='tab:blue',
                 #label='Cumulative precip. prev. year (mm)'
                 )
    ax1_twin.set_ylabel('Cumul. precip. prev. 90 days (mm)', color='tab:blue')
    ax1_twin.tick_params(axis='y', colors='tab:blue')
    
    # Plot the second DataFrame on the secondary y-axis
    sns.lineplot(x="date", y="tmean_mean_prev_7d_mean", 
                 data=df, ax=ax[2], color='tab:green',
                 #label='Mean temp. prev. year (°C)'
                 )
    ax[2].set_ylabel('Mean Temp. prev. 7 days (°C)', color='tab:green')
    ax[2].tick_params(axis='y', colors='tab:green')
    
    # Create a secondary y-axis
    ax2_twin = ax[2].twinx()
    
    sns.lineplot(x="date", y="precip_mean_prev_7d_sum", 
                 data=df, ax=ax2_twin, color='tab:blue',
                 #label='Cumulative precip. prev. 7 days (mm)'
                 )
    ax2_twin.set_ylabel('Cumul. precip. prev. 7 days (mm)', color='tab:blue')
    ax2_twin.tick_params(axis='y', colors='tab:blue')
    
    
    # Set the x-axis limits
    if start_date and end_date:
        plt.xlim(start_date, end_date)
    
    # Save the plot
    if save == True:
        plt.savefig('./figs/station_details.png', bbox_inches='tight')

    # Show the plot
    plt.show()
    
    
def plot_compare_scenario(past, base, pred):
    fig, axs = plt.subplots(4, 1, figsize=(9,7), sharex=True)
    for ax in axs:
        ax.invert_yaxis()
    
    sns.lineplot(data=past, x="date", y="water_depth", 
                 errorbar='sd', 
                 color='tab:orange', alpha=0.6, ax=axs[0],
                 label='Measured')
    sns.lineplot(data=past, x="date", y="pred_water_depth", 
                 errorbar='sd', 
                 color='tab:red', alpha=0.6, ax=axs[0],
                label='Predicted - original climate')
    axs[0].set_ylabel('Water depth (m)')
    
    sns.lineplot(data=base, x="date", y="pred_water_depth", 
                 errorbar='sd', 
                 color='tab:orange', alpha=0.6, ax=axs[1],
                 label='Measured')
    sns.lineplot(data=pred, x="date", y="pred_water_depth", 
                 errorbar='sd', 
                 color='tab:purple', alpha=0.6, ax=axs[1],
                label='Predicted - future climate')
    axs[1].set_ylabel('Water depth (m)')
    
    sns.lineplot(data=past, x="date", y="pred_water_depth", 
                 errorbar='sd', 
                 color='tab:red', alpha=0.6, ax=axs[2],
                 label='Predicted - original climate')
    sns.lineplot(data=pred, x="date", y="pred_water_depth", 
                 errorbar='sd', 
                 color='tab:purple', alpha=0.6, ax=axs[2],
                label='Predicted - future climate')
    axs[2].set_ylabel('Water depth (m)')
    
    sns.lineplot(data=pred, x="date", y="water_depth_anomaly", 
                 errorbar='sd', 
                 color='k', alpha=0.6, ax=axs[3],
                #label='Predicted - future climate'
                )
    axs[3].set_ylabel('Water depth anomaly (m)')
    axs[3].axhline(y=0, color='k', alpha=0.7)
    
    plt.tight_layout()
   
    
def plot_anomaly_map(df, date, dem, dem_extent, save=False):
    cmap = 'vlag'
    extent = df.geometry.total_bounds

    # Create a figure and axis with a PlateCarree projection
    fig, ax = plt.subplots(figsize=(8, 6), 
                           subplot_kw={'projection': ccrs.PlateCarree()} # removes axis labels
                          )
    
    # Plot hillshade DEM
    ax.imshow(dem, extent=(dem_extent[0], dem_extent[2],
                           dem_extent[1], dem_extent[3]),
              cmap='gray', origin='upper', aspect='auto')
    
    df.plot(column='water_depth_anomaly',
            cmap=cmap, norm=colors.CenteredNorm(halfrange=1.25),
            legend=True, ax=ax)
    
    ax.set_xlim(extent[0]-0.02, extent[2]+0.02)
    ax.set_ylim(extent[1]-0.02, extent[3]+0.02)
    
    #ax.legend(loc='lower left', framealpha=0.5)
    
    plt.title(f'Water depth anomaly (m) - {date}')

    if save == True:
        plt.savefig(f'./figs/anomaly_maps/{date}.png', bbox_inches='tight')
    
    # Show the plot
    plt.show()
    

def plot_measured_pred_map(df, date, dem, dem_extent, save=False):
    part = df.loc[df['date'] == date]
    extent = df.geometry.total_bounds
    
    fig = plt.figure(figsize=(10, 12))
    #fig.suptitle(f'{df["date"].dt.to_period("M")}')
    fig.suptitle(f"{date.strftime('%Y-%m-%d')}", fontsize=14, fontweight='bold')
    gs = GridSpec(5, 2, height_ratios=[5, 1, 1, 1, 1])
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[1, :]) 
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3) 
    ax5 = fig.add_subplot(gs[3, :], sharex=ax3) 
    ax6 = fig.add_subplot(gs[4, :], sharex=ax3) 
    
    ### ax1 ###
    ax1.imshow(dem, extent=(dem_extent[0], dem_extent[2],
                           dem_extent[1], dem_extent[3]),
              cmap='gray', origin='upper', aspect='auto')
    part.plot(column='water_depth',
            cmap='viridis', #norm=colors.CenteredNorm(halfrange=1.25),
            vmin=-0.5, vmax=10.5,
            legend=True, ax=ax1)
    ax1.set_xlim(extent[0]-0.02, extent[2]+0.02)
    ax1.set_ylim(extent[1]-0.02, extent[3]+0.02)
    ax1.set_title('Measured water depth (m)')
    
    ### ax2 ###
    ax2.imshow(dem, extent=(dem_extent[0], dem_extent[2],
                           dem_extent[1], dem_extent[3]),
              cmap='gray', origin='upper', aspect='auto')
    part.plot(column='residuals',
            cmap='vlag', norm=colors.CenteredNorm(halfrange=2),
            #norm=colors.SymLogNorm(linthresh=0.03, linscale=0.5,
            #                       vmin=-5, vmax=5, base=10),
            legend=True, ax=ax2)
    ax2.set_xlim(extent[0]-0.02, extent[2]+0.02)
    ax2.set_ylim(extent[1]-0.02, extent[3]+0.02)
    ax2.set_title('Residuals (m)')
    
    ### ax3 ###
    sns.lineplot(data=df, x="date", y="water_depth", 
                 errorbar='sd', 
                 color='tab:orange', alpha=0.6, ax=ax3,
                 label='Measured')
    sns.lineplot(data=df, x="date", y="pred_water_depth", 
                 errorbar='sd', 
                 color='tab:red', alpha=0.6, ax=ax3,
                label='Predicted')
    ax3.set_ylabel('Water depth (m)')
    ax3.invert_yaxis()
    ax3.axvline(x=date, color='k', alpha=0.2, linewidth=5)
    
    ### ax4 ###
    sns.lineplot(data=df, x="date", y="residuals", 
                 errorbar='sd', 
                 color='k', alpha=0.6, ax=ax4,
                #label='Predicted'
                )
    ax4.set_ylabel('Residuals (m)')
    ax4.invert_yaxis()
    ax4.axhline(y=0, color='k', alpha=0.4, linewidth=2)
    ax4.axvline(x=date, color='k', alpha=0.2, linewidth=5)
    
    ### ax5 ###
    ax5.axvline(x=date, color='k', alpha=0.2, linewidth=5)
    sns.lineplot(data=df, x="date", y="tmean_mean_prev_7d_mean", 
                 #errorbar='sd', 
                 color='tab:purple', alpha=0.6, ax=ax5,
                #label='T (°C)'
                )
    ax5.set_ylabel('T (°C) prev. 7d')
    
    ### ax6 ###
    ax6.axvline(x=date, color='k', alpha=0.2, linewidth=5)
    sns.lineplot(data=df, x="date", y="precip_mean_prev_7d_sum", 
                 #errorbar='sd', 
                 color='tab:cyan', alpha=0.6, ax=ax6,
                #label='T'
                )
    ax6.set_xlabel('Date')
    ax6.set_ylabel('P (mm) prev. 7d')
    
    for ax in [ax3, ax4, ax5]:
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_xlabel('')
    
    # Show the figure
    plt.tight_layout()
    #plt.show()
    
    if save == True:
        plt.savefig(f"./figs/past_plots/{date.strftime('%Y-%m-%d')}.png",
                    bbox_inches='tight')


def plot_pred_anomaly_map(df, date, dem, dem_extent, save=False):
    part = df.loc[df['date'] == date]
    extent = df.geometry.total_bounds
    
    fig = plt.figure(figsize=(10, 12))
    #fig.suptitle(f'{df["date"].dt.to_period("M")}')
    fig.suptitle(f"{date.strftime('%Y-%m-%d')}", fontsize=14, fontweight='bold')
    gs = GridSpec(3, 2, height_ratios=[3, 1, 1])
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[1, :]) 
    ax4 = fig.add_subplot(gs[2, :], sharex=ax3) 
    #ax5 = fig.add_subplot(gs[3, :], sharex=ax3) 
    #ax6 = fig.add_subplot(gs[4, :], sharex=ax3) 
    
    ### ax1 ###
    ax1.imshow(dem, extent=(dem_extent[0], dem_extent[2],
                           dem_extent[1], dem_extent[3]),
              cmap='gray', origin='upper', aspect='auto')
    part.plot(column='meas_water_depth',
            cmap='viridis', #norm=colors.CenteredNorm(halfrange=1.25),
            vmin=-0.5, vmax=10.5,
            legend=True, ax=ax1)
    ax1.set_xlim(extent[0]-0.02, extent[2]+0.02)
    ax1.set_ylim(extent[1]-0.02, extent[3]+0.02)
    ax1.set_title('Measured water depth (m)')
    
    ### ax2 ###
    ax2.imshow(dem, extent=(dem_extent[0], dem_extent[2],
                           dem_extent[1], dem_extent[3]),
              cmap='gray', origin='upper', aspect='auto')
    part.plot(column='water_depth_anomaly',
            cmap='vlag', norm=colors.CenteredNorm(halfrange=2),
            #norm=colors.SymLogNorm(linthresh=0.03, linscale=0.5,
            #                       vmin=-5, vmax=5, base=10),
            legend=True, ax=ax2)
    ax2.set_xlim(extent[0]-0.02, extent[2]+0.02)
    ax2.set_ylim(extent[1]-0.02, extent[3]+0.02)
    ax2.set_title('Water depth anomaly (m)')
    
    ### ax3 ###
    sns.lineplot(data=df, x="date", y="meas_water_depth", 
                 errorbar='sd', 
                 color='tab:orange', alpha=0.6, ax=ax3,
                 label='Measured')
    sns.lineplot(data=df, x="date", y="pred_water_depth", 
                 errorbar='sd', 
                 color='tab:red', alpha=0.6, ax=ax3,
                label='Predicted')
    ax3.set_ylabel('Water depth (m)')
    ax3.invert_yaxis()
    ax3.axvline(x=date, color='k', alpha=0.2, linewidth=5)
    
    ### ax4 ###
    sns.lineplot(data=df, x="date", y="water_depth_anomaly", 
                 errorbar='sd', 
                 color='k', alpha=0.6, ax=ax4,
                #label='Predicted'
                )
    ax4.set_ylabel('Anomaly (m)')
    ax4.invert_yaxis()
    ax4.axhline(y=0, color='k', alpha=0.4, linewidth=2)
    ax4.axvline(x=date, color='k', alpha=0.2, linewidth=5)
    
    for ax in [ax3]:#, ax4, ax5]:
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_xlabel('')
    
    # Show the figure
    plt.tight_layout()
    #plt.show()
    
    if save == True:
        plt.savefig(f"./figs/future_plots/{date.strftime('%Y-%m-%d')}.png",
                    bbox_inches='tight')
        

def create_year_df(year):
    '''year must be type int'''
    # Create a date range for the year with daily frequency
    date_range = pd.date_range(start=str(year) + '-01-01',
                               end=str(year) + '-12-31',
                               freq='D')
    
    # Create a DataFrame with the date range as the index
    df = pd.DataFrame(index=date_range)
    df['date'] = df.index
    # add column for merging
    df['month_day'] = df.index.strftime('%m-%d')

    return df


def merge_weather_dates(weather_df, year_df):
    synth = pd.merge(year_df, weather_df, 
                left_on='month_day', right_on='month_day',
                how='left',
                #left_index=True
                ).set_index(year_df['date'])

    #synth.index.rename('date')
    synth = synth.drop(['month_day', 'month', 'day'], axis=1)

    return synth


def create_weather(ref_year_df, start_year, running_year, 
                   y2y_temp_change, y2y_precip_change,
                  add_rand=True):
    ref_df = ref_year_df.copy()
    year_df = create_year_df(start_year + running_year)
    s = merge_weather_dates(ref_df, year_df)

    # calc long-term drift
    drift_precip = y2y_precip_change*running_year*s['precip_mean'].sum()/365
    drift_temp = y2y_temp_change*running_year

    # create short-term randomness
    if add_rand == True:
        # only add 0.5 std randomness to precip to prevent overestimation
        rand_precip = np.random.normal(loc=0, scale=s['precip_std']/2)
        rand_temp = np.random.normal(loc=0, scale=s['tmean_std'])
    else:
        rand_precip, rand_temp = 0, 0
    
    s['tmean_mean'] = s['tmean_mean'] + drift_temp + rand_temp
    
    s['precip_mean'] = s['precip_mean'] + drift_precip + rand_precip
    # prevent neg. precip. values
    s.loc[s['precip_mean'] < 0, 'precip_mean'] = 0
    
    return s


def create_resamples(past, pred, base, gs):
    pasts, preds, bases = {}, {}, {}
    intervals = {'weekly': 'W', 'monthly': 'M', 'quarterly': 'Q'}
    keep_cols = ['station_id',# 'geometry',
                 'reg_clusters',
                 'pred_water_depth',
                 'tmean_mean', 'precip_mean',
                 'tmean_mean_prev_7d_mean', 'precip_mean_prev_7d_sum',
                 'tmean_mean_prev_30d_mean', 'precip_mean_prev_30d_sum',
                 'tmean_mean_prev_1y_mean', 'precip_mean_prev_1y_sum']

    for i in intervals:
        pasts[i] = (past[keep_cols + ['water_depth', 'residuals']]
                    .groupby(by='station_id').resample(intervals[i]).mean())
        preds[i] = (pred[keep_cols]
                    .groupby(by='station_id').resample(intervals[i]).mean())
        bases[i] = (base[keep_cols]
                    .groupby(by='station_id').resample(intervals[i]).mean())

        pasts[i] = pasts[i].dropna()
        preds[i] = preds[i].dropna()
        bases[i] = bases[i].dropna()

        pasts[i].station_id = pasts[i].station_id.astype(int)
        preds[i].station_id = preds[i].station_id.astype(int)
        bases[i].station_id = bases[i].station_id.astype(int)

        pasts[i] = pasts[i].droplevel(0)
        preds[i] = preds[i].droplevel(0)
        bases[i] = bases[i].droplevel(0)
        
        pasts[i]['date'] = pasts[i].index
        preds[i]['date'] = preds[i].index
        bases[i]['date'] = bases[i].index

        pasts[i] = pd.merge(pasts[i], gs[['station_id', 'geometry']],
                            how='left', on='station_id')
        preds[i] = pd.merge(preds[i], gs[['station_id', 'geometry']],
                            how='left', on='station_id')
        bases[i] = pd.merge(bases[i], gs[['station_id', 'geometry']],
                            how='left', on='station_id')
        
        preds[i] = pd.merge(preds[i], pasts[i][['station_id', 'water_depth', 'date']],
                            left_on=['station_id', 'date'],
                            right_on=['station_id', 'date'],
                            how='left')
        preds[i] = preds[i].rename(columns={'water_depth': 'meas_water_depth'})
        
        #preds[i] = pd.merge(preds[i], bases[i][['station_id', 'pred_water_depth', 'date']],
        #                    left_on=['station_id', 'date'], right_on=['station_id', 'date'],
        #                    how='left')
        #preds[i] = preds[i].rename(columns={'pred_water_depth': 'base_water_depth'})
        
        #preds[i]['meas_water_depth'] = bases[i]['pred_water_depth']
        preds[i]['water_depth_anomaly'] = (preds[i]['pred_water_depth']
                                           - preds[i]['meas_water_depth'])

        pasts[i] = gpd.GeoDataFrame(pasts[i], geometry=pasts[i]['geometry'], crs='EPSG:4326')
        preds[i] = gpd.GeoDataFrame(preds[i], geometry=preds[i]['geometry'], crs='EPSG:4326')
        bases[i] = gpd.GeoDataFrame(bases[i], geometry=bases[i]['geometry'], crs='EPSG:4326')

    return pasts, preds, bases
