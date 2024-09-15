
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import folium
import geopandas as gpd
from folium.plugins import MarkerCluster
from folium.features import DivIcon
import plotly.express as px
import plotly.graph_objects as go
import contextily as ctx  # For adding basemaps
from shapely.geometry import Polygon, MultiPolygon
import gdown
from matplotlib.patches import Patch
import joblib
import ipywidgets as widgets
from IPython.display import display
import os
current_path = os.getcwd()
import streamlit as st

st.title("Prediksi Spasial DBD dan Kondisi Iklim")
st.write("""
Aplikasi ini memprediksi kemungkinan penyebaran Demam Berdarah Dengue (DBD) berdasarkan lokasi dan kondisi iklim.
Variabel kondisi iklim yang digunakan curah hujan, kelembaban, suhu, dan lama penyinaran matahari untuk mendapatkan hasil prediksi.
""")


# shp, dbf, shx, cpg
link_peta = ["https://drive.google.com/file/d/1wz21t9fHMPPUzMhYbnVKXmeynJA3_Q1R/view?usp=sharing",
             "https://drive.google.com/file/d/1d9Sl456szdpS_6Yu9KzWIKI990gVF6R1/view?usp=sharing",
             "https://drive.google.com/file/d/1TvDdouMACJB2egfMOmnpC4uyjNOYb_ig/view?usp=sharing",
             "https://drive.google.com/file/d/16Z_4tmqI4IXPMu2Iq4crjnOYmKGxBB-w/view?usp=sharing"]

FILE_ID = ["1wz21t9fHMPPUzMhYbnVKXmeynJA3_Q1R",
           "1d9Sl456szdpS_6Yu9KzWIKI990gVF6R1",
           "1TvDdouMACJB2egfMOmnpC4uyjNOYb_ig",
           "16Z_4tmqI4IXPMu2Iq4crjnOYmKGxBB-w"]

filetypes = ["shp", "dbf", "shx", "cpg"]
try:
  shapefile_path = current_path+'/lampung.shp'  # Update this path
except:
  for j in range(len(link_peta)):
    output = 'lampung.'+filetypes[j]
    gdown.download(f'https://drive.google.com/uc?export=download&id={FILE_ID[j]}', output, quiet=False)
  shapefile_path = current_path+'/lampung.shp'  # Update this path
gdf = gpd.read_file(shapefile_path)


# In[10]:


# Membaca data dari file CSV
link = "https://docs.google.com/spreadsheets/d/1no7J1H6IgYdQBLpXx4xhpA2bmCPOfQ1x/pub?output=csv"
data = pd.read_csv(link)

link2 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRam8x5Or7Mi7O60b-qXsi1HiEBrG5gzo3TWJLyFhfqNb3flo00SN0Lyxs4gX2n5T_cpETMNYLr-Tgn/pub?output=csv"
dataAsli = pd.read_csv(link2)

# # In[11]:

##=============Plot Data History 1
# fig, ax = plt.subplots(figsize=(10,4))
# ax.set_title('Kejadian DBD Kota Bandar Lampung 2009-2018\n dengan Relative Humidity')
# ax.plot(data['Incidence'])
# ax.set_xlabel('Bulan ke-')
# ax.set_ylabel('Insiden DBD')
# ax2 = ax.twinx()
# ax2.plot(data['RH_avg'], '--',color='tab:red')
# ax2.set_ylabel('Relative Humidity Avg. (%)', color='tab:red')
# ax2.tick_params(axis='y', labelcolor='tab:red')
# plt.show()



##=============Plot Data History 2
# fig, ax = plt.subplots(figsize=(10,4))
# ax.set_title('Kejadian DBD Kota Bandar Lampung 2009-2018\n dengan Curah Hujan')
# ax.plot(data['Incidence'])
# ax.set_xlabel('Bulan ke-')
# ax.set_ylabel('Insiden DBD')
# ax2 = ax.twinx()
# ax2.plot(data['RR'], '--',color='black')
# ax2.set_ylabel('Curah Hujan Avg. ', color='black')
# ax2.tick_params(axis='y', labelcolor='black')
# plt.show()


##=============Correlation 
# dat = data.drop(['Bulan', 'Tahun'], axis=1)
# corr_matrix = dat.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix Heatmap')
# plt.show()

dat = data.drop(['Bulan', 'Tahun'], axis=1)
corr_matrix = dat.corr()
incidence_corr = corr_matrix['Incidence']
selected_features = incidence_corr[abs(incidence_corr) > 0.1]
selected_features = selected_features[selected_features < 1]

selected_features_index = selected_features.index



# In[15]:


# Memastikan tidak ada nilai yang hilang
data = data.dropna()

# Memisahkan fitur dan target
X = data[selected_features_index]
y = data['Incidence']

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# In[20]:


try:
  model_path = current_path+"/random_forest_model.pkl"
  best_model = joblib.load(model_path)
  print("Model loaded successfully.")
except:
  param_grid = {
      'n_estimators': [10, 50, 100],
      'max_depth': [None, 2, 10, 20],
      'min_samples_split': [2, 5, 10, 20],
      'min_samples_leaf': [1, 2, 4, 8]
  }

  # Inisialisasi model Random Forest Regressor
  model = RandomForestRegressor()

  # Melatih model
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
  grid_search.fit(X_train, y_train)

  print("Best parameters:", grid_search.best_params_)
  print("Best score:", grid_search.best_score_)


  best_model = grid_search.best_estimator_
  # Save the trained model to a file
  joblib.dump(best_model, 'random_forest_model.pkl')


# In[21]:


y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R²: {r2}')


# In[ ]:





# In[22]:


y_pred_train = best_model.predict(X_train)

mse = mean_squared_error(y_train, y_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred_train)

print(f'RMSE: {rmse}')
print(f'R²: {r2}')


# In[23]:


# Plot prediksi vs realisasi
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Insiden DBD Aktual')
plt.ylabel('Insiden DBD Prediksi')
plt.title('Perbandingan Prediksi dan Aktual Insiden DBD')
plt.show()


# In[24]:


plt.plot(y_test.values, label='Data Aktual')
plt.plot(y_pred, label='Forecast')
plt.xlabel('Bulan')
plt.ylabel('Insiden DBD')
plt.legend()
plt.show()


# In[25]:


def calculate_dbd_incidence(number_of_cases, population):
    incidence_rate = (number_of_cases / population) * 100000
    return incidence_rate

def classify_incidence_rate(ai_values):
    classifications = []
    for ai in ai_values:
        if ai < 3:
            classifications.append("Aman")
        elif 3 <= ai < 10:
            classifications.append("Waspada")
        else:
            classifications.append("Awas")
    return classifications

def calculate_accuracy_and_create_table(actual_list, predicted_list):
    # Calculate accuracy
    accuracy = accuracy_score(actual_list, predicted_list)

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'Actual': actual_list,
        'Predicted': predicted_list
    })

    return accuracy, df



# In[26]:


# Example usage:
population_lampung = 1000000  # Example population of Lampung


dbd_incidence_rate = calculate_dbd_incidence(y_test.values, population_lampung)
print(dbd_incidence_rate)

predicted_incidence_rate = calculate_dbd_incidence(y_pred, population_lampung)
print(predicted_incidence_rate)


# In[27]:


dbd_incidence_class = classify_incidence_rate(dbd_incidence_rate)
print(dbd_incidence_class)

predicted_incidence_class = classify_incidence_rate(predicted_incidence_rate)
print(dbd_incidence_class)


# In[28]:


accuracy, df = calculate_accuracy_and_create_table(dbd_incidence_class, predicted_incidence_class)



# In[29]:


dbd_incidence_rate_train = calculate_dbd_incidence(y_train.values, population_lampung)
train_incidence_rate = calculate_dbd_incidence(y_pred_train, population_lampung)
print(dbd_incidence_rate_train)

dbd_incidence_rate_test = calculate_dbd_incidence(y_test.values, population_lampung)
test_incidence_rate = calculate_dbd_incidence(y_pred, population_lampung)


dbd_incidence_class_train = classify_incidence_rate(dbd_incidence_rate_train)
train_incidence_class = classify_incidence_rate(train_incidence_rate)


accuracy, df = calculate_accuracy_and_create_table(dbd_incidence_class_train, train_incidence_class)



# In[30]:


date_rng = pd.date_range('2009-03-01','2018-12-31', freq='MS')
date_train = date_rng[2:-22]
date_test = date_rng[-24:]



## Plot validation
# plt.subplots(figsize=(10,4))
# plt.plot(date_train, dbd_incidence_rate_train, label='data Train (aktual)')
# plt.plot(date_train, train_incidence_rate, label='Prediksi data Train')
# plt.plot(date_test, dbd_incidence_rate_test, label='data Test (aktual)')
# plt.plot(date_test, test_incidence_rate, label='Forecast')
# # plt.plot(y_predNN, label='Prediksi NN')
# # plt.axis([0, len(y_test), 0, 250])
# plt.title('Prediksi Kejadian DBD Kota Bandar Lampung 2009-2018\n dengan data Iklim')
# plt.xlabel('Bulan ke-')
# plt.ylabel('Insiden DBD (per 100.000 penduduk)')
# plt.xticks(date_rng[::4], [d.strftime('%b %Y') for d in date_rng[::4]], rotation=45)
# plt.legend()
# plt.show()


# In[32]:


color_map = {
    "Aman": "green",
    "Waspada": "yellow",
    "Awas": "red",
    "Unknown": "grey"
}

peluang_dict = {
    "Lampung Barat": 0.011680144,
    "Tanggamus": 0.09703504,
    "Lampung Selatan": 0.216531896,
    "Lampung Timur": 0.184186882,
    "Lampung Tengah": 0.097933513,
    "Lampung Utara": 0.095238095,
    "Way Kanan": 0.056603774,
    "Tulangbawang": 0.017070979,
    "Pesawaran": 0.208445642,
    "Pringsewu": 0.502246181,
    "Mesuji": 0.009883199,
    "Tulang Bawang Barat": 0.026954178,
    "Pesisir Barat": 0.004492363,
    "Kota Bandar Lampung": 1,
    "Kota Metro": 0.052111411
}


# In[33]:


k = 3
idxMonth = X_train.index[k]
st.write("Prediksi Bulan: ", data.iloc[idxMonth]['Bulan'], "Tahun ",  str(data.iloc[idxMonth]['Tahun']))
gdf['Status'] = 'Unknown'
gdf['AI'] = 'Tidak ada info'
gdf.loc[gdf['WADMKK'] == 'Kota Bandar Lampung', 'AI'] = np.round(dbd_incidence_rate_train[k],2)
gdf.loc[gdf['WADMKK'] == 'Kota Bandar Lampung', 'Status'] = train_incidence_class[k]
gdf['Color'] = gdf['Status'].map(color_map)
gdf.head()


# In[34]:
# Ensure the GeoDataFrame has a CRS (latitude and longitude)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

# Convert geometries to lat/lon for plotting with plotly
# Reproject to a suitable projected CRS (e.g., UTM zone 48N)
gdf_projected = gdf.to_crs(epsg=32648)

# Calculate centroids
gdf_projected['centroid'] = gdf_projected.centroid
gdf_projected['lon'] = gdf_projected.centroid.x
gdf_projected['lat'] = gdf_projected.centroid.y

# If needed, reproject centroids back to geographic CRS
gdf_centroid_geo = gdf_projected.to_crs(epsg=4326)
gdf['lon'] = gdf_centroid_geo.centroid.x
gdf['lat'] = gdf_centroid_geo.centroid.y

# Define color conditions based on 'AI' values
gdf['color'] = gdf['Color']

# Create Plotly figure
fig = go.Figure()

# Plot each polygon (area) with distinct borders, handling both Polygon and MultiPolygon
for idx, row in gdf.iterrows():
    geometry = row['geometry']
    
    if isinstance(geometry, Polygon):  # Single Polygon
        x, y = geometry.exterior.xy
        fig.add_trace(go.Scattermapbox(
            lon=list(x),  # Ensure it is a regular list
            lat=list(y),  # Ensure it is a regular list
            mode='lines',
            line=dict(width=2, color='black'),
            showlegend=False
        ))

    elif isinstance(geometry, MultiPolygon):  # MultiPolygon
        for poly in geometry.geoms:  # Iterate over each polygon in the MultiPolygon using `.geoms`
            x, y = poly.exterior.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(x),  # Ensure it is a regular list
                lat=list(y),  # Ensure it is a regular list
                mode='lines',
                line=dict(width=2, color='black'),
                showlegend=False
            ))

# Plot centroids with AI values (always visible)
for idx, row in gdf.iterrows():
    fig.add_trace(go.Scattermapbox(
        lon=[row['lon']],
        lat=[row['lat']],
        mode='text',
        text=[f"AI: {row['AI']}"],
        textfont=dict(size=12, color="black"),
        showlegend=False
    ))

# Add map layout with OpenStreetMap tiles
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=8,
    mapbox_center={"lat": -5.4291, "lon": 105.2610},  # Center on Lampung
    height=600,
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

# Add a custom legend manually using annotations
fig.add_trace(go.Scattermapbox(
    lon=[None], lat=[None],
    mode='markers',
    marker=dict(size=10, color='green'),
    name='Aman: AI < 3'
))
fig.add_trace(go.Scattermapbox(
    lon=[None], lat=[None],
    mode='markers',
    marker=dict(size=10, color='yellow'),
    name='Waspada: 3 ≤ AI < 10'
))
fig.add_trace(go.Scattermapbox(
    lon=[None], lat=[None],
    mode='markers',
    marker=dict(size=10, color='red'),
    name='Bahaya: AI ≥ 10'
))
# Show the plot (in Jupyter) or display in Streamlit using `st.plotly_chart()`
st.plotly_chart(fig)

st.write("""
**Informasi Kondisi Iklim:**
         """)
st.write("- Suhu minimum bulan " + data.iloc[idxMonth-1]['Bulan'], ' ', str(data.iloc[idxMonth-1]['Tahun']),': ', str(data.iloc[idxMonth-1]['Tn-1']))
st.write("- Suhu maksimum bulan " + data.iloc[idxMonth-1]['Bulan'], ' ', str(data.iloc[idxMonth-1]['Tahun']),': ', str(data.iloc[idxMonth-1]['Tx-1']))
st.write("- Suhu rerata bulan " + data.iloc[idxMonth-1]['Bulan'], ' ', str(data.iloc[idxMonth-1]['Tahun']),': ', str(data.iloc[idxMonth-1]['Tavg-1']))
st.write("- Curah hujan bulan " + data.iloc[idxMonth-1]['Bulan'], ' ', str(data.iloc[idxMonth-1]['Tahun']),': ', str(data.iloc[idxMonth-1]['RR-1']))
st.write("- Lama penyinaran matahari bulan " + data.iloc[idxMonth-1]['Bulan'], ' ', str(data.iloc[idxMonth-1]['Tahun']),': ', str(data.iloc[idxMonth-1]['ss-1']))
st.write("- Kejarian DBD bulan " + data.iloc[idxMonth-1]['Bulan'], ' ', str(data.iloc[idxMonth-1]['Tahun']),': ', str(data.iloc[idxMonth-1]['I-1']))

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: lightgray;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        &copy; 2024 | Made with ❤️ by <a href="https://github.com/fauzirifky" target="_blank">fauzirifky</a>
        | Supported by <a href="https://lppm.itera.ac.id" target="_blank">LPPM Itera</a>
            </div>
    """, unsafe_allow_html=True)
st.write("Sistem ini bertujuan untuk memberikan prediksi awal berdasarkan data iklim. Hasil yang diperoleh terbatas pada Kota Bandar Lampung karena alasan ketersediaan data.")
