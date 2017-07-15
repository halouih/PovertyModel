import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
import prim
import os

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
mylabels = ['Share of workers in the agriculture sector', 'Share of workers in the manufacture sector', 'Employment rate', 'Productivity growth in the services sector', 'Productivity growth in the agriculture sector', 'Productivity growth in the manufacture sector', 'Skill premium in the services sector', 'Skill premium in the agriculture sector', 'Skill premium in the manufacture sector', 'Share of national income used to pay for pensions', 'Share of national income used for redistribution', 'Low population growth and high education (SSP5 population)', 'Voice and governance', 'Climate change impacts']
relabel = dict(zip(myinputs, mylabels))


wbccodes = pd.read_csv("wbccodesHamza.csv")
ini_data = pd.read_csv("ini_data_info_may14.csv")

for i in wbccodes.index:
    wbccodes.loc[i,'country'] = correct_countrycode(wbccodes.loc[i,'country'])

nameofthisround = 'sept2016_4'
model = os.getcwd()
bau_folder   = "{}/baselines_{}/".format(model,nameofthisround)
cc_folder    = "{}/with_cc_{}/".format(model,nameofthisround)
    
    
list_bau = os.listdir(bau_folder)
list_cc  = os.listdir(cc_folder)

list_country = []
for i in list_bau:
    list_country.append(i[:3])

for selectedcountry in list_country:

    
    bau = pd.DataFrame()
    cc = pd.DataFrame()


    country_name = wbccodes.loc[wbccodes.country==selectedcountry].country_name.tolist()[0]

    bau = pd.read_csv(bau_folder+selectedcountry+'_bau.csv')
    cc = pd.read_csv(cc_folder+selectedcountry+'_cc.csv')
    bau = bau.dropna()
    cc = cc.dropna()

    bau['issp5'] = bau.ssp=='ssp5'
    cc['issp5'] = cc.ssp=='ssp5'


    bau_c = bau[bau.country==selectedcountry]
    hip = bau_c
    cc_c = cc[cc.country==selectedcountry]
    hop = cc_c


    ini_data['countryname'] = ini_data.countrycode.replace(wbccodes.set_index('country').country_name)
    
    hop['ini_incbott40'] = hop.countryname.replace(ini_data.set_index('countryname').incbott40)
    hop['ini_avincome'] = hop.countryname.replace(ini_data.set_index('countryname').avincome)
    hip['ini_incbott40'] = hip.countryname.replace(ini_data.set_index('countryname').incbott40)
    hip['ini_avincome'] = hip.countryname.replace(ini_data.set_index('countryname').avincome)

    hop['shprosp']=hop['incbott40']/hop['ini_incbott40']*hop['ini_avincome']/hop['avincome']
    hip['shprosp']=hip['incbott40']/hip['ini_incbott40']*hip['ini_avincome']/hip['avincome']

    hop['shprospdiff'] = hop['shprosp']-hip['shprosp']
    hop['incbott20diff'] = (hop.incbott20-hip.incbott20)/hip.incbott20
    hop['avincomediff'] = (hop.avincome-hip.avincome)/hip.avincome
    hop['ginidiff'] = hop['gini']-hip['gini']
    
    

    # ----------------------------------BASELINE---------------------------------------------------------
    
    
    oois = ['incbott20','shprosp']

    data = np.array([[hip[oois[0]][0],hip[oois[1]][0]]])
    for i in range(1,len(hip.index)):
        data = np.concatenate((data,np.array([[hip[oois[0]][i],hip[oois[1]][i]]])))

    data_norm = normalize(data)

    #Initial Values

    n_clusters = 4
    f_value = 0.5
    seuil = 1.2
    #Clustering: Gaussian Mixture Model. It returns a dataframe with normalized incbott20 and shared prosperity along with the corresponding class label

    def clustering(n_clusters):
        g = mixture.GMM(n_components=n_clusters)
        g.fit(data_norm)
        pred = g.predict(data_norm)+1
                    
        x=[]
        y=[]
        for i in range(0,len(data)):
            x.append(data[i][0])
            y.append(data[i][1])

        df = pd.DataFrame()
        df['incbott20'] = x
        df['shprosp'] = y
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
            if 1 in boxes[i].peeling_trajectory['coverage']:
                coverage1 = np.where(boxes[i].peeling_trajectory['coverage']==1)[0][0]
                obj = obj.drop(obj.index[[coverage1]])
            k = obj.argmin()
            boxes[i].select(k)
    
        return [boxes,df]

    # The following loop determins the maximum number of clusters under 4 that have the best coverage+density
    # 'condition' is a variable that is True if we are out of the loop without decrementing the number of clusters
    # in which case, we must incremente the number of clusters if we are out of the loop after a decrementation

    while n_clusters>2:
    
        condition = True
    
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


    #Plot of clusters


    dummy = get_prim(n_clusters, f_value)
    boxes = dummy[0]
    df = dummy[1]

    classes = df[['class']].drop_duplicates()['class']

    fig = plt.figure(figsize=(10,10))
    plt.xlabel('Average income of the bottom 20% ($/month)', fontsize=20)
    plt.ylabel('Shared prosperity: Av. income of the bottom 40% / Av. income', fontsize=20)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)

    for i in range(1,max(classes)+1):
        to_plot_x = df.ix[df['class'] == i].incbott20
        to_plot_y = df.ix[df['class'] == i].shprosp
    
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
    
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = pylatex.Document(geometry_options=geometry_options)

    


    with doc.create(pylatex.Section(country_name+" baseline")):
        
        plt.title("Cluster Distributions")
        with doc.create(pylatex.Figure()) as graph:
            graph.add_plot()
    
        doc.append(pylatex.VerticalSpace("10pt"))
    
        label2 = 1
        
        for i in range(0,len(boxes)):
            with doc.create(pylatex.Figure()) as graph1:
                boxes[i].inspect(style='graph')
                plt.title("Box "+str(label2))
                plt.tight_layout()
                graph1.add_plot(width = '10cm')
                label2 = label2 + 1
            doc.append(pylatex.VerticalSpace("2pt"))
        
        with doc.create(pylatex.Tabular(table_spec='|c|c|')) as table1:
            table1.add_hline()
            for j in relabel:
                table1.add_row([j, relabel[j]])
                table1.add_hline()
                
    doc.append(pylatex.NewPage())


# ----------------------------------CLIMATE CHANGE----------------------------------------------------------
    
    
    oois = ['incbott20','shprosp']

    data = np.array([[hop[oois[0]][0],hop[oois[1]][0]]])
    for i in range(1,len(hop.index)):
        data = np.concatenate((data,np.array([[hop[oois[0]][i],hop[oois[1]][i]]])))

    data_norm = normalize(data)

    #Initial Values

    n_clusters = 4
    f_value = 0.5
    seuil = 1.2
    #Clustering: Gaussian Mixture Model. It returns a dataframe with normalized incbott20 and shared prosperity along with the corresponding class label

    def clustering(n_clusters):
        g = mixture.GMM(n_components=n_clusters)
        g.fit(data_norm)
        pred = g.predict(data_norm)+1
                    
        x=[]
        y=[]
        for i in range(0,len(data)):
            x.append(data[i][0])
            y.append(data[i][1])

        df = pd.DataFrame()
        df['incbott20'] = x
        df['shprosp'] = y
        df['class'] = pred
        df['issp5'] = hop['issp5']
    
        return df



    #Prim Algorithm

    def get_prim(n_clusters, f_value):
    
        df = clustering(n_clusters)
        classes = df[['class']].drop_duplicates()['class']
        list_prim =[]
        sorted(classes)
        
        for i in range(1,max(classes)+1):
            list_prim.append(prim.Prim(hop[myinputs], (df['class']==i),threshold=0.5,threshold_type=">"))

        boxes = []

        for i in range(0,len(list_prim)):
            boxes.append(list_prim[i].find_box())
            obj = (f_value*boxes[i].peeling_trajectory['coverage']-(1-f_value)*boxes[i].peeling_trajectory['density'])**2
            if 1 in boxes[i].peeling_trajectory['coverage']:
                coverage1 = np.where(boxes[i].peeling_trajectory['coverage']==1)[0][0]
                obj = obj.drop(obj.index[[coverage1]])
            k = obj.argmin()
            boxes[i].select(k)
    
        return [boxes,df]

    # The following loop determins the maximum number of clusters under 4 that have the best coverage+density
    # 'condition' is a variable that is True if we are out of the loop without decrementing the number of clusters
    # in which case, we must incremente the number of clusters if we are out of the loop after a decrementation

    while n_clusters>2:
    
        condition = True
    
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


    #Plot of clusters


    dummy = get_prim(n_clusters, f_value)
    boxes = dummy[0]
    df = dummy[1]

    classes = df[['class']].drop_duplicates()['class']

    fig = plt.figure(figsize=(10,10))
    plt.xlabel('Average income of the bottom 20% ($/month)', fontsize=20)
    plt.ylabel('Shared prosperity: Av. income of the bottom 40% / Av. income', fontsize=20)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)

    for i in range(1,max(classes)+1):
        to_plot_x = df.ix[df['class'] == i].incbott20
        to_plot_y = df.ix[df['class'] == i].shprosp
    
        plt.subplot()
        plt.scatter(to_plot_x, to_plot_y, c = color[i-1])


    recs = []
    classes_b = []
    for i in range(0,len(classes)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=color[i]))
        classes_b.append('Box '+str(i+1))
    plt.legend(recs,classes_b,loc=1)


    # LATEX


    


    with doc.create(pylatex.Section(country_name+" with climate change")):
        
        plt.title("Cluster Distributions")
        with doc.create(pylatex.Figure()) as graph:
            graph.add_plot()
    
        doc.append(pylatex.VerticalSpace("10pt"))
    
        label2 = 1
        
        for i in range(0,len(boxes)):
            with doc.create(pylatex.Figure()) as graph1:
                boxes[i].inspect(style='graph')
                plt.title("Box "+str(label2))
                plt.tight_layout()
                graph1.add_plot(width = '10cm')
                label2 = label2 + 1
            doc.append(pylatex.VerticalSpace("2pt"))
        
        with doc.create(pylatex.Tabular(table_spec='|c|c|')) as table1:
            table1.add_hline()
            for j in relabel:
                table1.add_row([j, relabel[j]])
                table1.add_hline()
                
    doc.append(pylatex.NewPage())
                
    
    # ----------------------------------DIFFERENCE CLIMATE CHANGE BASELINE---------------------------
    
    
    oois = ['incbott20diff','shprospdiff']

    data = np.array([[hop[oois[0]][0],hop[oois[1]][0]]])
    for i in range(1,len(hop.index)):
        data = np.concatenate((data,np.array([[hop[oois[0]][i],hop[oois[1]][i]]])))

    data_norm = normalize(data)

    #Initial Values

    n_clusters = 4
    f_value = 0.5
    seuil = 1.2
    #Clustering: Gaussian Mixture Model. It returns a dataframe with normalized incbott20 and shared prosperity along with the corresponding class label

    def clustering(n_clusters):
        g = mixture.GMM(n_components=n_clusters)
        g.fit(data_norm)
        pred = g.predict(data_norm)+1
                    
        x=[]
        y=[]
        for i in range(0,len(data)):
            x.append(data[i][0])
            y.append(data[i][1])

        df = pd.DataFrame()
        df['incbott20diff'] = x
        df['shprospdiff'] = y
        df['class'] = pred
    
        return df



    #Prim Algorithm

    def get_prim(n_clusters, f_value):
    
        df = clustering(n_clusters)
        classes = df[['class']].drop_duplicates()['class']
        list_prim =[]
        sorted(classes)
        
        for i in range(1,max(classes)+1):
            list_prim.append(prim.Prim(hop[myinputs], (df['class']==i),threshold=0.5,threshold_type=">"))

        boxes = []

        for i in range(0,len(list_prim)):
            boxes.append(list_prim[i].find_box())
            obj = (f_value*boxes[i].peeling_trajectory['coverage']-(1-f_value)*boxes[i].peeling_trajectory['density'])**2
            if 1 in boxes[i].peeling_trajectory['coverage']:
                coverage1 = np.where(boxes[i].peeling_trajectory['coverage']==1)[0][0]
                obj = obj.drop(obj.index[[coverage1]])
            k = obj.argmin()
            boxes[i].select(k)
    
        return [boxes,df]

    # The following loop determins the maximum number of clusters under 4 that have the best coverage+density
    # 'condition' is a variable that is True if we are out of the loop without decrementing the number of clusters
    # in which case, we must incremente the number of clusters if we are out of the loop after a decrementation

    while n_clusters>2:
    
        condition = True
    
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


    #Plot of clusters


    dummy = get_prim(n_clusters, f_value)
    boxes = dummy[0]
    df = dummy[1]

    classes = df[['class']].drop_duplicates()['class']

    fig = plt.figure(figsize=(10,10))
    plt.xlabel('Relative loss of average income of the bottom 20% due to climate change', fontsize=20)
    plt.ylabel('Loss of shared prosperity due to climate change', fontsize=20)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)

    for i in range(1,max(classes)+1):
        to_plot_x = df.ix[df['class'] == i].incbott20diff
        to_plot_y = df.ix[df['class'] == i].shprospdiff
    
        plt.subplot()
        plt.scatter(to_plot_x, to_plot_y, c = color[i-1])


    recs = []
    classes_b = []
    for i in range(0,len(classes)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=color[i]))
        classes_b.append('Box '+str(i+1))
    plt.legend(recs,classes_b,loc=1)


    # LATEX


    with doc.create(pylatex.Section(country_name+" difference between climate change and baseline")):
        
        plt.title("Cluster Distributions")
        with doc.create(pylatex.Figure()) as graph:
            graph.add_plot()
    
        doc.append(pylatex.VerticalSpace("10pt"))
    
        label2 = 1
        
        for i in range(0,len(boxes)):
            with doc.create(pylatex.Figure()) as graph1:
                boxes[i].inspect(style='graph')
                plt.title("Box "+str(label2))
                plt.tight_layout()
                graph1.add_plot(width = '10cm')
                label2 = label2 + 1
            doc.append(pylatex.VerticalSpace("2pt"))
        
        with doc.create(pylatex.Tabular(table_spec='|c|c|')) as table1:
            table1.add_hline()
            for j in relabel:
                table1.add_row([j, relabel[j]])
                table1.add_hline()
    
    
    
    
    
    
    
    
    
    doc.generate_pdf(model+'/2DAnalysis/'+selectedcountry)
    
    print(selectedcountry+" is added")
            

