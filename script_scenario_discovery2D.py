import scipy.stats as stats
import matplotlib.ticker as ticker
from matplotlib import cm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from sklearn.decomposition import PCA
import prim
import os
from decimal import Decimal

sns.set_style('white')

def normalize(data):
    minima = np.min(data, axis=0)
    maxima = np.max(data, axis=0)
    a = 1/(maxima-minima)
    b = minima/(minima-maxima)
    data = a * data + b                    
    return data

def arrondi(df):
    dg = df
    for i in df.index:
        for j in df.columns:
            if j == 'qp values':
                dg.loc[i,j] = "{:.2e}".format(df.loc[i,j])
            else:
                try:
                    dg.loc[i,j] = round(df.loc[i,j],2)
                except:
                    pass
    return dg

def correct_countrycode(countrycode):
    '''
    Corrects countrycodes in the database that don't correspond to official 3 letters codes.
    '''
    if countrycode=='TMP':
        countrycode='TLS'
    if countrycode=='ZAR':
        countrycode='COD'
    if countrycode=='ROM':
        countrycode='ROU'
    return countrycode

color = ['b','g','r','c']
myinputs = ['shareag','sharemanu', 'shareemp', 'grserv', 'grag', 'grmanu', 'skillpserv','skillpag', 'skillpmanu', 'p', 'b','issp5','voice','ccint']
selectedcountry = 'MAR'

wbccodes = pd.read_csv("wbccodesHamza.csv")

for i in wbccodes.index:
    wbccodes.loc[i,'country'] = correct_countrycode(wbccodes.loc[i,'country'])

nameofthisround = 'sept2016_4'
model = os.getcwd()
bau_folder   = "{}/baselines_{}/".format(model,nameofthisround)


list_bau = os.listdir(bau_folder)
bau = pd.DataFrame()

country_name = wbccodes.loc[wbccodes.country==selectedcountry].country_name.tolist()[0]

bau = pd.read_csv(bau_folder+selectedcountry+'_bau.csv')
bau = bau.dropna()

bau['issp5'] = bau.ssp=='ssp5'


bau_c = bau[bau.country==selectedcountry]

hip = bau_c
hip['shared_prosperity'] = (hip['incbott40']/hip['avincome'])
oois = ['incbott20','shared_prosperity']

data = np.array([[hip[oois[0]][0],hip[oois[1]][0]]])
for i in range(1,len(hip.index)):
    data = np.concatenate((data,np.array([[hip[oois[0]][i],hip[oois[1]][i]]])))

data = normalize(data)

#Initial Values

n_clusters = 4
f_value = 0.5
seuil = 1.3

#Clustering: Gaussian Mixture Model. It returns a dataframe with normalized incbott20 and shared prosperity along with the corresponding class label

def clustering(n_clusters):
    g = mixture.GMM(n_components=n_clusters, n_iter=500)
    g.fit(data)
    pred = g.predict(data)+1
    class_outcomes = {'classes':pred}
                    
    x=[]
    y=[]
    for i in range(0,len(data)):
        x.append(data[i][0])
        y.append(data[i][1])

    df = pd.DataFrame()
    df['incbott20'] = x
    df['shared_prosperity'] = y
    df['class'] = pred
    
    return df



#Prim Algorithm

def get_prim(n_clusters, f_value):
    
    df = clustering(n_clusters)
    classes = df[['class']].drop_duplicates()['class']
    list_prim =[]
    sorted(classes)

    for i in range(1,max(classes)+1):
        list_prim.append(prim.Prim(hip[myinputs], (df['class']==i),threshold=0.5,threshold_type=">"))

    boxes = []

    for i in range(0,len(list_prim)):
        boxes.append(list_prim[i].find_box())
        obj = (f_value*boxes[i].peeling_trajectory['coverage']-(1-f_value)*boxes[i].peeling_trajectory['density'])**2
        k = np.where(obj==np.min(obj))[0][0]
        boxes[i].select(k)
    
    return [boxes,df]

# The following loop determins the maximum number of clusters under 4 that have the best coverage+density

condition = True
# 'condition' is a variable that is True if we are out of the loop without decrementing the number of clusters
# in which case, we must incremente the number of clusters if we are out of the loop after a decrementation

while n_clusters>2:
    for i in get_prim(n_clusters, f_value)[0]:
        if i.coverage+i.density < seuil:
            condition = False
        else:
            condition = condition&True
    if condition == True:
        break
    else:
        n_clusters = n_clusters - 1
        continue


if condition == False:
    n_clusters = n_clusters+1


#Plot of clusters


dummy = get_prim(n_clusters, f_value)
boxes = dummy[0]
df = dummy[1]

classes = df[['class']].drop_duplicates()['class']

fig = plt.figure(figsize=(10,10))
plt.xlabel('Average income of the bottom 20% (normalized)', fontsize=20)
plt.ylabel('Shared Prosperity (normalized)', fontsize=20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)

for i in range(1,max(classes)+1):
    to_plot_x = df.ix[df['class'] == i].incbott20
    to_plot_y = df.ix[df['class'] == i].shared_prosperity
    
    plt.subplot()
    plt.scatter(to_plot_x, to_plot_y, c = color[i-1])


import matplotlib.patches as mpatches
recs = []
classes_b = []
for i in range(0,len(classes)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=color[i]))
    classes_b.append('Box '+str(i+1))
plt.legend(recs,classes_b,loc=1)


# LATEX

import pylatex

if selectedcountry == 'MAR':
    
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = pylatex.Document(geometry_options=geometry_options)

    


    with doc.create(pylatex.Section(country_name)):
        
        plt.title("Cluster Distributions")
        with doc.create(pylatex.Figure()) as graph:
            graph.add_plot()
    
        doc.append(pylatex.VerticalSpace("10pt"))
    
        label2 = 1
        
        with doc.create(pylatex.Subsection("Restricted Dimensions")):
            for i in range(0,len(boxes)):
                with doc.create(pylatex.Figure()) as graph1:
                    boxes[i].inspect(style='graph')
                    plt.title("Box "+str(label2))
                    plt.tight_layout()
                    graph1.add_plot(width = '10cm')
                    label2 = label2 + 1
                doc.append(pylatex.VerticalSpace("2pt"))
        


    doc.generate_tex(model+'/2DAnalysis/'+selectedcountry)
            

