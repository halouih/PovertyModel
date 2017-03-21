# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:28:31 2017

@author: Hamza
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from lib_for_growth_model import *
from decimal import Decimal
import numpy as np
from scipy.stats import gaussian_kde
from pandas_helper import *
from sklearn import tree
import pydotplus

from lib_for_plot import *
import seaborn as sns

sns.set_context("poster",rc={"font.size": 18})
sns.set_style("whitegrid")
import matplotlib.gridspec as gridspec

codes = read_csv('wbccodes2014.csv')
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
	
from lib_for_analysis import *


codes['country'] = codes.country.apply(correct_countrycode)

nameofthisround = 'sept2016_separate_impacts'
model = os.getcwd()
bau_folder   = "{}/baselines_{}/".format(model,nameofthisround)
cc_folder    = "{}/with_cc_{}/".format(model,nameofthisround)


list_bau = os.listdir(bau_folder)
list_cc  = os.listdir(cc_folder)
bau = pd.DataFrame()
cc  = pd.DataFrame()

selectedcountry = 'MAR'

bau = pd.read_csv(bau_folder+'bau_'+selectedcountry+'.csv').drop('Unnamed: 0',axis=1)
print('cc_'+selectedcountry+'.csv')
cc = pd.read_csv(cc_folder+'cc_'+selectedcountry+'.csv').drop('Unnamed: 0',axis=1)

bau = bau.drop_duplicates(['country', 'scenar', 'ssp'])
cc = cc.dropna()
bau = bau.dropna()

for col_switch in ['switch_ag_rev','switch_temp', 'switch_ag_prices', 'switch_disasters', 'switch_health']:
    cc.ix[cc[col_switch],"switch"] = col_switch

cc['issp5'] = cc.ssp=='ssp5'
bau['issp5'] = bau.ssp=='ssp5'
bau['countryname'] = bau.country.replace(codes.set_index('country').country_name)
cc['countryname'] = cc.country.replace(codes.set_index('country').country_name)


bau_c = bau[bau.country==selectedcountry]
cc_c  = cc[cc.country==selectedcountry]

hop = cc_c.set_index(['scenar', 'ssp','ccint','switch'])
hip = broadcast_simple(bau_c.set_index(['scenar', 'ssp']),hop.index)

hop['below125diff'] = (hop['below125']-hip['below125'])
hop['below125diff_pc_of_pop'] = (hop['below125']-hip['below125'])/hop['tot_pop']
hop['below125diff_relative_change'] = (hop['below125']-hip['below125'])/hip['below125']


hop['incbott40diff'] = (hop.incbott40-hip.incbott40)/hip.incbott40
hop['avincomediff'] = (hop.avincome-hip.avincome)/hip.avincome


titles = ["Agriculture\n revenues","Labor\nproductivity","Food prices","Disasters","Health"]

myinputs = ['shareag','sharemanu', 'shareemp', 'grserv', 'grag', 'grmanu', 'skillpserv','skillpag', 'skillpmanu', 'p', 'b','issp5','voice']

clf1 = tree.DecisionTreeRegressor(criterion='mse', max_depth=2, min_samples_leaf=int(hop[myinputs].shape[0]*0.05))
clf = clf1.fit(hop[myinputs], hop['incbott40diff'])
list_boxes,threevar,threesign,threevalues=get_lineage(clf, myinputs)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=myinputs, class_names=['pc in extr pov'], filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(selectedcountry+"ExtremePoverty.pdf")

sample = []
income = []

for i in range(0,len(list_boxes)):
    sample.append(hop[eval(list_boxes[i])])
    income.append(sample[i]['incbott40diff'])

for a in income:
    sns.distplot(hop['incbott40diff'])

