import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# %matplotlib inline
import time
import seaborn as sns
import math
from scipy.spatial.distance import euclidean, chebyshev, cityblock
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import unidecode
from unidecode import unidecode
import requests
import pandas as pd
import datetime
# %matplotlib inline
from dateutil.parser import parse
# from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from statistics import mean
from statistics import stdev

# driver = webdriver.Chrome()
# driver.get('https://www.google.com/flights/explore/')
# time.sleep(6)

def scrape_data(start_date, from_place, to_place, city_name):
    driver = webdriver.Chrome()
    driver.get('https://www.google.com/flights/explore/')
    time.sleep(6)

    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()
    to_action = ActionChains(driver)
    to_action.send_keys(to_place)
    to_action.send_keys(Keys.ENTER)
    to_action.perform()
    
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]/div/div')
    from_input.click()
    from_action = ActionChains(driver)
    from_action.send_keys(from_place)
    from_action.send_keys(Keys.ENTER)
    from_action.perform() 
    
    time.sleep(5)
    current_url = driver.current_url
    split_url = current_url.split('2017')
    new_url = split_url[0] + start_date
    driver.get(new_url)    
    
    time.sleep(5)
    results = driver.find_elements_by_class_name('LJTSM3-v-c')
    

    result_names = []
    for i in range(len(results)):
        result_names.append(results[i].text)

    cities = []
    for city in result_names:
        name = city.split(',')
        cities.append(unidecode(name[0]))
    
    lower_case = [x.lower() for x in cities]
    if city_name.lower() in lower_case:
        city_index = lower_case.index(city_name.lower())
    
    time.sleep(2)
    
    box_class = driver.find_elements_by_class_name("LJTSM3-v-m")
    test = box_class[city_index]
    bars = test.find_elements_by_class_name('LJTSM3-w-x')

    time.sleep(1)

    data = []

    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.00001)
        data.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
           test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))

    time.sleep(2)

    d = data[0]
    clean_data = [(float(d[0].replace('$', '').replace(',', '')), (parse(d[1].split('-')[0].strip()) - datetime.datetime(2017,4,4,0,0)).days)
                      for d in data]

    time.sleep(3)

    df = pd.DataFrame(clean_data, columns=['Price','Start_Date'])
    return df

# df = scrape_data('2017-04-16','New York','United States','Miami')
# print df

def scrape_data_90(start_date, from_place, to_place, city_name):
    
    driver = webdriver.Chrome()
    driver.get('https://www.google.com/flights/explore/')
    time.sleep(6)
    
    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()
    to_action = ActionChains(driver)
    to_action.send_keys(to_place)
    to_action.send_keys(Keys.ENTER)
    to_action.perform()
    
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]/div/div')
    from_input.click()
    from_action = ActionChains(driver)
    from_action.send_keys(from_place)
    from_action.send_keys(Keys.ENTER)
    from_action.perform() 
    
    time.sleep(5)
    current_url = driver.current_url
    split_url = current_url.split('2017')
    new_url = split_url[0] + start_date
    driver.get(new_url)    
    
    time.sleep(5)
    results = driver.find_elements_by_class_name('LJTSM3-v-c')
    

    result_names = []
    for i in range(len(results)):
        result_names.append(results[i].text)

    cities = []
    for city in result_names:
        name = city.split(',')
        cities.append(unidecode(name[0]))
    
    lower_case = [x.lower() for x in cities]
    if city_name.lower() in lower_case:
        city_index = lower_case.index(city_name.lower())
    
    time.sleep(2)
    
    box_class = driver.find_elements_by_class_name("LJTSM3-v-m")
    test = box_class[city_index]
    bars = test.find_elements_by_class_name('LJTSM3-w-x')

    time.sleep(1)
    data_90 = []

    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.00001)
        data_90.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
           test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))
    
    move_bars_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[4]/div/div[2]/div[1]/div/div[2]/div[2]/div/div[2]/div[5]')
    move_bars_input.click()
    move_bars_input = ActionChains(driver)
    move_bars_input.perform()
    
    time.sleep(5)
    results = driver.find_elements_by_class_name('LJTSM3-v-c')
    result_names = []
    for i in range(len(results)):
        result_names.append(results[i].text)

    cities = []
    for city in result_names:
        name = city.split(',')
        cities.append(unidecode(name[0]))
        
    lower_case = [x.lower() for x in cities]
    if city_name.lower() in lower_case:
        city_index = lower_case.index(city_name.lower())
        
    time.sleep(2)
        
    box_class = driver.find_elements_by_class_name("LJTSM3-v-m")
    test = box_class[city_index]
    bars = test.find_elements_by_class_name('LJTSM3-w-x')
    
    next_data= []
    for bar in bars:
        ActionChains(driver).move_to_element(bar).perform()
        time.sleep(0.00001)
        next_data.append((test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,
           test.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))
    
    time.sleep(3)
    for element in next_data:
        if element not in data_90:
            data_90.append(element)
            
    d = data_90[0]
    clean_data = [(float(d[0].replace('$', '').replace(',', '')), (parse(d[1].split('-')[0].strip()) - datetime.datetime(2017,4,4,0,0)).days)
                      for d in data_90]
    
    df_90 = pd.DataFrame(clean_data, columns=['Price','Start_Date'])
    return df_90   
    
# df_90 = scrape_data_90('2017-04-20','London','Europe','Dublin')
# print 90

def task_3_dbscan(flight_data):
       
    px = [x for x in df['Price']]
    ff = pd.DataFrame(px, columns=['Price_of_flight']).reset_index()

    X = MinMaxScaler().fit_transform(ff) 
    db = DBSCAN(eps = 0.05, min_samples = 3).fit(X)

    labels = db.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize = (12,8))

    for k,c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor = c, markeredgecolor = 'k',markersize = 14)
    
    plt.title('Total Clusters: {}'.format(clusters), fontsize = 14, y = 1.01) 
    plt.savefig('task_3_dbscan.png')

    def calculate_cluster_means(X, labels):
        lbls = np.unique(labels)
        print "Cluster labels: {}".format(np.unique(lbls))

        cluster_means = [np.mean(X[labels==num, :], axis=0) for num in range(lbls[-1] + 1)]
        print "Cluster Means: {}".format(cluster_means)
        return cluster_means

    cluster_means = calculate_cluster_means(X, labels)

    out_ind = []
    for ind,y in enumerate(X):
        if labels[ind]== -1:
            out_ind.append([ind,y])

    class_member_mask = (labels == -1) 
    outs = X[class_member_mask]

    chosen_list = []

    for out_index,out in enumerate(outs):    
        min_list = []
        for index,cm in enumerate(cluster_means):         
            min_list.append(euclidean(cm, out))

        for inx,abc in enumerate(min_list):
            if abc == sorted(min_list)[0]:
                chosen_list.append([out_index,inx])
            
    outlier_prices = []
    for i,abc in enumerate(out_ind):
        outlier_prices.append(df['Price'][abc[0]])
    
    def cluster_mean_price(index):
        clus_ind = []
        for ind,y in enumerate(X):
            if labels[ind]== index:
                clus_ind.append([ind,y])
    
        class_member_mask_clus = (labels == index) 
        clus = X[class_member_mask_clus] 
    
        clus_prices=[]
        for i,abc in enumerate(clus_ind):
            clus_prices.append(df['Price'][abc[0]])   

        return clus_prices
    
    
    mean_of_chosenlist_clusters= []
    for b in chosen_list:
        mean_of_chosenlist_clusters.append(mean(cluster_mean_price(b[1])))
    
    stdev_of_chosenlist=[]
    for b in chosen_list:
        stdev_of_chosenlist.append(stdev(cluster_mean_price(b[1])))
    
    outlier_indexes = []
    for b in chosen_list:
        outlier_indexes.append(b[0])

    
    df_best_price = pd.DataFrame(columns=('Start_Date','Price'))
    best_price_count = 0
    for x in range(0, (len(outlier_indexes))-1):    
        m = int(mean_of_chosenlist_clusters[x])
        s = int(stdev_of_chosenlist[x])
        if (outlier_prices[x] <  (m - (2 * s))) and  (outlier_prices[x] < (m - 50)):
            best_price_count += 1
            df_best_price.loc[best_price_count] = df.loc[outlier_indexes[x]]
        
    return df_best_price

def task_3_IQR(flight_data):    
    df['Price'].plot.box()
    plt.savefig('task_3_IQR.png')
    
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    
#     i = 'Price'
#     ax = df[i].plot(kind='kde')
#     plt.subplot(212)
#     plt.xlim(df[i].min(), df[i].max()*1.1)
#     sns.boxplot(x=df[i])
#     plt.axvline(x=min)
#     plt.axvline(x=max)
#     plt.savefig('task_3_IQR.png')

    iqr_outlier_index = []
    iqr_outlier_data = []

    for i, x in enumerate(df['Price']):
        if (x < abs(Q1 - (1.5 * IQR))) or (x > abs(Q3 + (1.5 * IQR))):
            iqr_outlier_index.append(i)

    for x in iqr_outlier_index:
        iqr_outlier_data.append([df['Start_Date'][x], df['Price'][x]])

    return pd.DataFrame(iqr_outlier_data, columns=['Start_Date', 'Price'])


