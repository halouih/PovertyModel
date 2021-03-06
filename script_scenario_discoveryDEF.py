"""

Determination of optimal clusters, density & coverage

@author: Hamza
"""

import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import prim 
import seaborn as sns
import pylatex
import matplotlib.pyplot as plt


wbccodes = pd.read_csv("wbccodesHamza.csv")

nameofthisround = 'sept2016_4'
model = os.getcwd()
bau_folder   = "{}/baselines_{}/".format(model,nameofthisround)
cc_folder    = "{}/with_cc_{}/".format(model,nameofthisround)


list_bau = os.listdir(bau_folder)
list_cc  = os.listdir(cc_folder)
bau = pd.DataFrame()
cc  = pd.DataFrame()

def arrondi(df):
    dg = df
    for i in df.index:
        for j in df.columns:
            try:
                dg.loc[i,j] = round(df.loc[i,j],2)
            except:
                pass
    return dg

def report(selectedcountry):
    
    country_name = wbccodes.loc[wbccodes.country==selectedcountry].country_name.tolist()[0]
    
    plt.figure(figsize=(12,5))
    sns.set_context("poster",rc={"font.size": 18})
    sns.set_style("whitegrid")

    bau = pd.read_csv(bau_folder+selectedcountry+'_bau.csv')
    print(selectedcountry+'_cc.csv')
    cc = pd.read_csv(cc_folder+selectedcountry+'_cc.csv')
    cc = cc.dropna()
    bau = bau.dropna()

    for col_switch in ['switch_ag_rev','switch_temp', 'switch_ag_prices', 'switch_disasters', 'switch_health']:
        cc.ix[cc[col_switch],"switch"] = col_switch

    cc['issp5'] = cc.ssp=='ssp5'
    bau['issp5'] = bau.ssp=='ssp5'


    bau_c = bau[bau.country==selectedcountry]
    cc_c  = cc[cc.country==selectedcountry]

    hop = cc_c
    hip = bau_c

    hop['below125diff'] = (hop['belowpovline']-hip['belowpovline'])
    hop['below125diff_pc_of_pop'] = (hop['belowpovline']-hip['belowpovline'])/hop['tot_pop']
    hop['below125diff_relative_change'] = (hop['belowpovline']-hip['belowpovline'])/hip['belowpovline']


    hop['incbott40diff'] = (hop.incbott40-hip.incbott40)/hip.incbott40
    hop['avincomediff'] = (hop.avincome-hip.avincome)/hip.avincome


    titles = ["Agriculture\n revenues","Labor\nproductivity","Food prices","Disasters","Health"]

    myinputs = ['shareag','sharemanu', 'shareemp', 'grserv', 'grag', 'grmanu', 'skillpserv','skillpag', 'skillpmanu', 'p', 'b','issp5','voice','ccint']



    # Initial number of clusters
    clusters = 4
    # Threshold for coverage+density of clustering
    seuil = 1.3
    # Weight of density/coverage
    f_value = 0.35

    # The following function extracts the boxes from the clustering and the respective stats
    def monk(clusters, seuil, f_value):

        kmeans = KMeans(n_clusters=clusters).fit(hop[['incbott40diff']])
    
        l = []

        for i in range(0,clusters):
            l.append(kmeans.cluster_centers_[i][0])

        sorted_centers = np.array(l)
        sorted_centers.sort()

        nodes = []

        for i in range(0,clusters-1):
            nodes.append((sorted_centers[i]+sorted_centers[i+1])/2)


        list_prim = []

        list_prim.append(prim.Prim(hop[myinputs],(hop.incbott40diff<nodes[0]),threshold=0.5,threshold_type=">"))

        for i in range(0,len(nodes)-1):
            list_prim.append(prim.Prim(hop[myinputs],(hop.incbott40diff>nodes[i])&(hop.incbott40diff<nodes[i+1]),threshold=0.5,threshold_type=">"))

        list_prim.append(prim.Prim(hop[myinputs],(hop.incbott40diff>nodes[len(nodes)-1]),threshold=0.5,threshold_type=">"))

        boxes = []

        for i in range(0,len(list_prim)):
            boxes.append(list_prim[i].find_box())
            obj = f_value*boxes[i].peeling_trajectory['coverage']+(1-f_value)*boxes[i].peeling_trajectory['density']
            k = np.where(obj==np.max(obj))[0][0]
            boxes[i].select(k)
                      
        return boxes


    # The following loop determins the maximum number of clusters under 4 that have the best coverage+density

    condition = True
    # 'condition' is a variable that is True if we are out of the loop without decrementing the number of clusters
    # in which case, we must incremente the number of clusters if we are out of the loop after a decrementation

    while clusters>2:
        for i in monk(clusters, seuil, f_value):
            if i.coverage+i.density < seuil:
                condition = False
            else:
                condition = condition&True
        if condition == True:
            break
        else:
            clusters = clusters - 1
            continue


    if condition == False:
        clusters = clusters+1

    boxes = []

    def get_boxes(box):
        dfs = []
        for i in box:
            data = i.limits
            list_index = data.index
            df = hop
        
            for j in list_index:
                if j == 'issp5':
                    if True in data.loc[j,'min']:
                        df = df[(df[j] == True)]
                    else:
                        df = df[(df[j] == False)]
                
                else:
                    df = df[(df[j]>=data.loc[j,'min'])&(df[j]<=data.loc[j,'max'])]
            
            dfs.append(df)
        return dfs



    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = pylatex.Document(geometry_options=geometry_options)

    label = 1
    for i in get_boxes(monk(clusters, seuil, f_value)):
        sns.distplot(i['incbott40diff'], axlabel = False, label = 'Box '+str(label))
        plt.legend()
        label = label+1


    with doc.create(pylatex.Section(country_name)):
        
        plt.title("Cluster Distributions")
        with doc.create(pylatex.Figure()) as graph:
            graph.add_plot()
    
        doc.append(pylatex.VerticalSpace("10pt"))
    
        label2 = 1
        for i in monk(clusters, seuil, f_value):
            limits = i.limits
            limits = arrondi(limits)
            stats = pd.DataFrame.from_dict(data=i.stats, orient='index')
            stats.columns = ['value']
            stats = arrondi(stats)
            
            doc.append('Box'+str(label2))
            doc.append(pylatex.LineBreak())
            with doc.create(pylatex.Tabular(table_spec='|c|c|')) as table1:
                table1.add_hline()
                table1.add_row(['',stats.columns[0]])
                table1.add_hline()
                for j in stats.index:
                    table1.add_row([j, stats.loc[j][0]])
                    table1.add_hline()
        
            doc.append(pylatex.HorizontalSpace("10pt"))
            
            with doc.create(pylatex.Tabular(table_spec='|c|c|c|c|')) as table2:
                table2.add_hline()
                table2.add_row(['',limits.columns[0],limits.columns[1],limits.columns[2]])
                table2.add_hline()
                for j in limits.index:
                    table2.add_row([j, limits.loc[j][0],limits.loc[j][1],limits.loc[j][2]])
                    table2.add_hline()
            
            doc.append(pylatex.VerticalSpace("10pt"))
            doc.append(pylatex.LineBreak())
            
            label2 = label2+1


    doc.generate_tex(model+'/ScenarDisc/'+selectedcountry)
            
