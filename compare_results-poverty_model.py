

import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn import mixture
import prim
import os

folder = os.getcwd()

wbccodes = pd.read_csv(folder+"/wbccodes.csv")
ssp_gdp = pd.read_csv(folder+"/SspDb_country_data_2013-06-12.csv")
ini_data = pd.read_csv(folder+"/ini_data_info_may14.csv")
codes_tables = pd.read_csv(folder+"/ISO3166_and_R32.csv")
ini_year = 2007


def get_gdp_growth(ssp_data,year,ssp,r32,ini_year):
    model='OECD Env-Growth'
    scenario="SSP{}_v9_130325".format(ssp)
    selection=(ssp_data['MODEL']==model)&(ssp_data['SCENARIO']==scenario)&(ssp_data['REGION']==r32)&(ssp_data['VARIABLE']=='GDP|PPP')

    if ini_year<2010:
        y1=ssp_data.ix[selection,'2005'].values[0]
        y2=ssp_data.ix[selection,'2010'].values[0]
        f=interpolate.interp1d([2005,2010], [y1,y2],kind='slinear')
    else:
        y1=ssp_data.ix[selection,'2010'].values[0]
        y2=ssp_data.ix[selection,'2015'].values[0]
        f=interpolate.interp1d([2010,2015], [y1,y2],kind='slinear')

    gdp_ini=f(ini_year)
    gdp_growth=ssp_data.ix[selection,str(year)].values[0]/gdp_ini
    return gdp_growth


def country2r32(codes_tables,countrycode):
    r32='R32{}'.format(codes_tables.loc[codes_tables['ISO']==countrycode,'R32'].values[0])
    return r32

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

def normalize(data):
    minima = np.min(data, axis=0)
    maxima = np.max(data, axis=0)
    a = 1/(maxima-minima)
    b = minima/(minima-maxima)
    data = a * data + b                    
    return data


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


def future_gdp_ssp5(cc):
    future_gdp_ssp5 = ini_data.ix[ini_data['countrycode']==cc,'gdp']*get_gdp_growth(ssp_gdp,2030,5,cc,ini_year)
    return future_gdp_ssp5

def future_gdp_ssp4(cc):
    future_gdp_ssp4 = ini_data.ix[ini_data['countrycode']==cc,'gdp']*get_gdp_growth(ssp_gdp,2030,4,cc,ini_year)
    return future_gdp_ssp4
    


# THE FOLLOWING PERFORMS THE CLUSTERING-----------------------------------------------------------------------------------



color = ['b','g','r','c']
myinputs = ['shareag','sharemanu', 'shareemp', 'grserv', 'grag', 'grmanu', 'skillpserv','skillpag', 'skillpmanu', 'p', 'b','issp5','voice','ccint']
mylabels = ['Share of workers in the agriculture sector', 'Share of workers in the manufacture sector', 'Employment rate', 'Productivity growth in the services sector', 'Productivity growth in the agriculture sector', 'Productivity growth in the manufacture sector', 'Skill premium in the services sector', 'Skill premium in the agriculture sector', 'Skill premium in the manufacture sector', 'Share of national income used to pay for pensions', 'Share of national income used for redistribution', 'Low population growth and high education (SSP5 population)', 'Voice and governance', 'Climate change impacts']
relabel = dict(zip(myinputs, mylabels))
results = pd.DataFrame(columns=myinputs+['future_gdp','model_gdp','country'])



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
    
    
    
    oois = ['incbott20','shprosp']

    data = np.array([[hop[oois[0]][0],hop[oois[1]][0]]])
    for i in range(1,len(hop.index)):
        data = np.concatenate((data,np.array([[hop[oois[0]][i],hop[oois[1]][i]]])))

    data_norm = normalize(data)

    #Initial Values

    n_clusters = 4
    f_value = 0.5
    seuil = 1.2

    

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

    df['issp5'] = hop['issp5']



    # ------------------------- Scenario determination SSP5 ------------------------------------------------------


    # 1) We select a cluster, we will choose first the highest average income of the bottom 20% cluster
    # 2) We only keep the data points that are described by the Prim Box
    # 3) We minimize the following criteria: opt = (outcomes['GDP']/gdp2compare-1)**2

    #si le cluster 1 n'a pas des scenarios avec la pop du SSP5 tu prends le deuxieme
    inichoice = []
    for i in range(0,n_clusters):
        inichoice.append(sum((df['class']==i+1)&(df['issp5'])))

    ssp5_cluster = np.argmax(inichoice) + 1


    print("Cluster "+ str(i) +" selected")

    # NUAGE DE POINTS, SELECTION

    hop['class'] = df['class']
    selection = hop.ix[hop['class']==ssp5_cluster]

    temp = boxes[ssp5_cluster-1].limits
    for j in temp.index:
        if j == 'issp5':
            if temp.loc[j,'min']=={True}:
                selection = selection.ix[selection[j]==True]
            else:
                selection = selection.ix[selection[j]==False]
        else:
            selection = selection.ix[selection[j]>=temp.loc[j,'min']]
            selection = selection.ix[selection[j]<=temp.loc[j,'max']]


    opt = (selection.ix[(selection['issp5']==True)&(selection['class']==ssp5_cluster)]['GDP']/future_gdp_ssp5(selectedcountry).iloc[0]-1)**2
    scenario = opt.argmin()

    to_add = hop.iloc[[scenario]][myinputs]
    to_add['future_gdp'] = future_gdp_ssp5(selectedcountry).values
    to_add['model_gdp'] = hop.ix[[scenario],'GDP'].values
    to_add['country'] = [selectedcountry]
    results.loc[len(results)] = to_add.loc[scenario]
    
    print(selectedcountry+" IS NOW DONE")
    


results.to_csv(folder+'Results.csv', index=False)
