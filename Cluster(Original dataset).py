
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
#import tensorflow as tf
#assert tf.__version__ >= "2.0"
#from tensorflow import keras

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.grid(False)
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "data"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan as NA


Rock_property = pd.read_csv('G:\工作项目资料文件夹\中英文论文写作\广安小论文\广安小论文3\\Rock_data_2020.csv',
                       encoding = "gb18030")
print(Rock_property.head())

Rock_use_name = ['Density','water_content','Por','CS','tensile_strength',
                 'deformation_modulus','elastic_modulus',"poisson_ratio",
                 "soften_coefficient","Top"] 
Rock_property_use = Rock_property[Rock_use_name]
print(Rock_property_use.info()) 
print(Rock_property_use.head())


Property_trans = Rock_property_use.copy()

Property_trans["deformation_modulus"] = pd.to_numeric(Property_trans["deformation_modulus"] ,
                                                      errors='coerce')
Property_trans["elastic_modulus"] = pd.to_numeric(Property_trans["elastic_modulus"] ,
                                                  errors='coerce')

df_des = Property_trans.describe()
print(Property_trans.describe())
df_des.to_excel('G:\工作项目资料文件夹\中英文论文写作\广安小论文\广安小论文3\\原始数据统计信息.xlsx', sheet_name='Sheet1')
print(Property_trans.info())

print(Property_trans[Property_trans.isnull().values ==True])

Property_trans.to_excel('G:\工作项目资料文件夹\中英文论文写作\广安小论文\广安小论文3\\数据转换后结果.xlsx', sheet_name='Sheet1')

Property_trans.hist(bins=30,figsize = (20,15))
plt.show()

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 40,
    'text.color': 'black'
})

Pro_prepared = Property_trans.copy()

corr_matrix = Pro_prepared.corr()
print(corr_matrix["Top"].sort_values(ascending=False))
from pandas.plotting import scatter_matrix

attributes = ["Top","CS", 'tensile_strength',"Density", "Por"]
              
scatter_matrix(Pro_prepared[attributes], figsize=(20,15))
save_fig("scatter_matrix_plot")

import seaborn as sns

dict_s = {'粉质粘土':0,'砂岩':1,'泥岩':3,'泥质粉砂岩':2,'粉砂质泥岩':4}


#sns.set_style('white')
scatter_use_name = ['Density','water_content','Por','CS','tensile_strength',
                 'deformation_modulus','elastic_modulus',"poisson_ratio",
                 "soften_coefficient","Top",'Lithos']

scatter = Rock_property[scatter_use_name]
scatter['Lithos'] = scatter['Lithos'].replace(dict_s)

sns.set(rc={"xtick.direction":"in", "ytick.direction":"in"})
sns.set_style({'axes.axisbelow': True, 'axes.edgecolor': '.8', 'axes.linewidth': 0.5, 'xtick.direction': 'in', 
               'ytick.direction': 'in', 'xtick.major.size': 3.0, 'ytick.major.size': 3.0}) 
sns.set_style("ticks",{"xtick.direction":"in", "ytick.direction":"in"})       


sns.pairplot(scatter)

sns.pairplot(scatter, hue ='Lithos',palette='rainbow',diag_kind='kde',plot_kws={'alpha':0.3})

plt.figure(figsize=(12, 8))
save_fig("pairplot.png")


plt.figure(figsize=(12, 8))
sns.heatmap(scatter.corr(), annot=True, cmap="YlGnBu");
save_fig("heatmap.png")
plt.show()

Pro_prepared.drop('Top',axis =1,inplace = True) 


mean = Pro_prepared.mean().values
std = Pro_prepared.std().values

print(mean)
print(std)
print(type(mean))


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())])

Pro_scaler = num_pipeline.fit_transform(Pro_prepared)


Pro_scaler1 = np.round(Pro_scaler,3) 
print(Pro_scaler1)



from sklearn.decomposition import PCA
scatter_use_name = ['Density','water_content','Por','CS','tensile_strength',
                 'deformation_modulus','elastic_modulus',"poisson_ratio",
                 "soften_coefficient","Top",'Lithos']

df_p = pd.DataFrame(Pro_scaler1.copy()) 
df_p.to_csv('Pro_scaler1.csv', index=False)

P1 = df_p[[1,2]]
P2 = df_p[[3,4,5,6,7,8]]
pca1 = PCA(n_components=1)
pca2 = PCA(n_components=2)

P1_pca = pca1.fit_transform(P1)
P2_pca = pca2.fit_transform(P2)

df_p['P1'] = pd.DataFrame(P1_pca, index=P1.index)
df_p[['P2a','P2b']] = pd.DataFrame(P2_pca, index=P2.index)


Pro_scaler1 = df_p[[0,'P1','P2a','P2b']]




from sklearn.cluster import KMeans
kmeans_per_k = [KMeans(n_clusters=k, random_state=42,
                       init='k-means++').fit(Pro_scaler1) for k in range(2, 15)] 
                 
inertias = [model.inertia_ for model in kmeans_per_k]
plt.figure(figsize=(12, 8))
plt.plot(range(2, 15), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(5, inertias[4]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.axis([1, 15, 0, 5000])
save_fig("inertia_vs_k_plot")
plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = [silhouette_score(Pro_scaler1, model.labels_)
                     for model in kmeans_per_k[1:]]
plt.figure(figsize=(12, 8))
plt.plot(range(2, 14), silhouette_scores, "bo-") 
plt.xlabel("$k$", fontsize=14) 
plt.ylabel("Silhouette score", fontsize=14) 
plt.axis([1, 14, 0.3, 0.6]) 
save_fig("silhouette_score_vs_k_plot")
plt.show()

from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6, 7, 8): 
    plt.subplot(2, 3, k - 2)
    
    y_pred1 = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(Pro_scaler1, y_pred1)

    padding = len(Pro_scaler1) // 30 
    pos = padding
    ticks = []
    for i in range(k): 
        coeffs = silhouette_coefficients[y_pred1 == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in range(3, 6): 
        plt.ylabel("Cluster",fontsize=14)
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
    if k in range(6, 9): 
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient",fontsize=14)
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

save_fig("silhouette_analysis_plot")
plt.show()

from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred2 = kmeans.fit_predict(Pro_scaler1)
print(y_pred2) 
print(y_pred2.shape)
print(y_pred2 is kmeans.labels_) 
print(kmeans.cluster_centers_ ) 

center = pd.DataFrame(kmeans.cluster_centers_)

print(center) 

Rock_property["classfication"]=pd.DataFrame(y_pred2)
Rock_property.to_excel("G:\工作项目资料文件夹\中英文论文写作\广安小论文\广安小论文3\\原始数据分类结果.xlsx")

import plotly.graph_objects as go
import plotly as py


df = Rock_property.copy()
df['classfication'] = pd.DataFrame(y_pred2)

print(df['classfication'])



data=[go.Parcoords(
        line = dict(color = df['classfication'],
                   colorscale = [[0,'#D7C16B'],[0.5,'#23D8C3'],[1,'#F3F10F']
                                 ]),
        dimensions = list([
            dict(range = [2.7,2],
                constraintrange = [2.7,2], 
                label = 'Density', values = df.iloc[:,5], 
                tickvals = [round(x,2) for x in [2 + i * 0.07 for i in range(11)]]),
            
            dict(range = [15,0],
                label = 'water_content', values = df.iloc[:,6],
                tickvals = [round(x,2) for x in [0.0 + i * 1.5 for i in range(11)]]),
            
            dict(range = [30,0],
                label = 'Por', values = df.iloc[:,7],
                tickvals = [round(x,2) for x in [0.0 + i * 3 for i in range(11)]]),
            
            dict(range = [0,70],  
                label = 'CS', values = df.iloc[:,8],
                tickvals = [round(x,2) for x in [0 + i * 7 for i in range(11)]]),        
            
            dict(range = [0,4.7],
                label = 'tensile_strength', values = df.iloc[:,9],
                tickvals = [round(x,2) for x in [0 + i * 0.47 for i in range(11)]]),
            
            dict(range = [420,15000],
                label = 'deformation_modulus', values = df.iloc[:,10],
                tickvals = [round(x,2) for x in [420 + i * 1458 for i in range(11)]]),
            
            dict(range = [400,16000],
                label = 'elastic_modulus', values = df.iloc[:,11],
                tickvals = [round(x,2) for x in [400 + i * 1560 for i in range(11)]]),
            
            dict(range = [0,0.5],
                label = 'poisson_ratio', values = df.iloc[:,12],
                tickvals = [round(x,2) for x in [0 + i * 0.05 for i in range(11)]]),
            
            dict(range = [0,1],
                label = 'soften_coefficient', values = df.iloc[:,13],
                tickvals = [round(x,2) for x in [0 + i * 0.1 for i in range(11)]]),
            
            dict(range = [0,4],
                label = 'classfication', values = df['classfication'],
                tickvals = [0,1,2,3,4]),            
        ])
    )]

layout = go.Layout(
                   font=dict(family = 'Times New Roman',size = 30,color ='black'),
                   width=1600,
                   height=900)
fig = go.Figure(data=data, layout=layout)

py.offline.plot(fig, filename = '原始数据.html')
               


df1 = pd.DataFrame(Pro_scaler1.copy()) 
df1['classfication'] = pd.DataFrame(y_pred2)
print(df1['classfication'])
print(df1.columns)
df1.to_excel("G:\工作项目资料文件夹\中英文论文写作\广安小论文\广安小论文3\\特征缩放分类结果.xlsx")
data=[go.Parcoords(
        line = dict(color = df1['classfication'],
                   colorscale = [[0,'#D7C16B'],[0.5,'#23D8C3'],[1,'#F3F10F']
                                 ]),
        dimensions = list([
            dict(range = [-4,2],
                 constraintrange = [-4,2],
                label = 'Density', values = df1.iloc[:,0],
                tickvals = [round(x,2) for x in [-4 + i * 0.6 for i in range(11)]]),
            dict(range = [-4,5],
                label = 'P1', values = df1.iloc[:,1],
                tickvals = [round(x,2) for x in [-4 + i * 0.9 for i in range(11)]]),
            dict(range = [-4,6],
                label = 'P2a', values = df1.iloc[:,2],
                tickvals = [round(x,2) for x in [-4 + i * 1 for i in range(11)]]),
            dict(range = [-4,3], 
                label = 'P2b', values = df1.iloc[:,3],
                tickvals = [round(x,2) for x in [-4 + i * 0.7 for i in range(11)]]),
            dict(range = [0,4],
                label = 'classfication', values = df['classfication'],
                tickvals = [0,1,2,3,4]),          
        ])
    )]
layout = go.Layout(
                   font=dict(family = 'Times New Roman',size = 30,color ='black'),
                   width=1600,
                   height=900)
fig = go.Figure(data=data, layout=layout)

py.offline.plot(fig, filename = '特征缩放数据.html')
               

print(Rock_property.groupby('classfication').Lithos.count())

df2 = Rock_property.groupby('classfication').Lithos.value_counts()
print(Rock_property.groupby('classfication').Lithos.value_counts())
df2.to_excel("G:\工作项目资料文件夹\中英文论文写作\广安小论文\广安小论文3\\分类结果岩性数量.xlsx")


