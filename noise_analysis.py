import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

##les dic1 et dic2 sont obtenus en faisant runner le code lireJson_bruit sur les clusters 
path1 = r'C:\Users\laura\Documents\Master 2\Stage St Luc\Notebook_for_clustering\dic1.json'
path2 = r'C:\Users\laura\Documents\Master 2\Stage St Luc\Notebook_for_clustering\dic2.json'
dic1 = json.load(open(path1))
dic2 = json.load(open(path2))

patients = ["01","02","04","05","08","09","11","12","13","14","15","17","18","19","20",
            "21","22","24","26","27","28","29","30","31","32","33","34","35","36","37",
            "39","40","41","42","43","45","46","48","50","51","52","53"]

                    
def dataframe_mvt(dic1, dic2) : 
    """
    Création d'un dataframe avec les colonnes suivantes : 
        Patient, time, qc_cnr_avg, qc_cnr_std, qc_mot_abs, qc_mot_rel
    """
    dic_inter = {}
    lst_good = []
    for i in dic1.keys():
        number1 = i[3:5]
        dic_inter[number1] = 1
    for j in dic2.keys():
        number2 = j[3:5]
        if number2 in dic_inter.keys():
            dic_inter[number2] = 2
    for key, value in dic_inter.items():
        if value == 2:
            lst_good.append(key)
    lst_good.sort()
    
    index = 0
    time_lst = [1, 2]
    X     = len(lst_good)*len(time_lst)
    col_1 = np.zeros(X)
    col1  = pd.DataFrame(col_1, columns=['patient'])
    col_2 = np.zeros(X)
    col2  = pd.DataFrame(col_2, columns=['time'])
    col_3 = np.zeros(X)
    col3  = pd.DataFrame(col_3, columns=['qc_cnr_avg'])
    col_4 = np.zeros(X)
    col4  = pd.DataFrame(col_4, columns=['qc_cnr_std'])
    col_5 = np.zeros(X)
    col5  = pd.DataFrame(col_5, columns=['qc_mot_abs'])
    col_6 = np.zeros(X)
    col6  = pd.DataFrame(col_6, columns=['qc_mot_rel'])
    df    = pd.concat([col1, col2, col3, col4, col5, col6], axis = 1)
       
    for patient_nb in lst_good: 
        for E in time_lst: 
            key = "sub"+ patient_nb + "_E" + str(E) 
            if E == 1 : 
                dic_rep = dic1[key]
                df['patient'][index]    = dic_rep['Patients']
                df['time'][index]       = dic_rep['Time']
                df['qc_cnr_avg'][index] = dic_rep['CNR_avg']
                df['qc_cnr_std'][index] = dic_rep['CNR_std']
                df['qc_mot_abs'][index] = dic_rep['Movement_abs']
                df['qc_mot_rel'][index] = dic_rep['Movement_rel']
            if E == 2 : 
                dic_rep = dic2[key]
                df['patient'][index]    = dic_rep['Patients']
                df['time'][index]       = dic_rep['Time']
                df['qc_cnr_avg'][index] = dic_rep['CNR_avg']
                df['qc_cnr_std'][index] = dic_rep['CNR_std']
                df['qc_mot_abs'][index] = dic_rep['Movement_abs']
                df['qc_mot_rel'][index] = dic_rep['Movement_rel']
            index+=1         
    return df

def graph(df, param, patient_numbers, clusters, colors):
    """
    Plot des paramètres suivants : 
        qc_cnr_avg, qc_cnr_std, qc_mot_abs, qc_mot_rel
    avec en abscisse : 1 pour temps1 et 2 pour temps2
    
    Chaque couleur représente un cluster 
    Chaque pointillé représente un patient 
    Chaque trait continu représente la moyenne du paramètre sur tout le cluster
    """
    x = [1, 2]
    mean_0 = [0, 0]
    mean_1 = [0, 0]
    mean_2 = [0, 0]
    plt.figure()
    for patient_nb in patient_numbers:
        params = []
        for i in range (len(df)): 
            if df['patient'][i] == int(patient_nb): 
                if df['time'][i] == 1 : 
                    params.append(df[param][i])
                if df['time'][i] == 2  : 
                    params.append(df[param][i])
                    if patient_nb in clusters[0]: 
                        mean_0[0] += params[0]
                        mean_0[1] += params[1]
                        plt.scatter(x, params, color = colors[0])
                        plt.plot(x, params,'--', color = colors[0])
                    if patient_nb in clusters[1]: 
                        mean_1[0] += params[0]
                        mean_1[1] += params[1]
                        plt.scatter(x, params, color = colors[1])
                        plt.plot(x, params,'--',  color = colors[1])
                    if patient_nb in clusters[2]: 
                        mean_2[0] += params[0]
                        mean_2[1] += params[1]
                        plt.scatter(x, params, color = colors[2])
                        plt.plot(x, params, '--', color = colors[2])
    mean_0[0] /= len(clusters[0])
    mean_0[1] /= len(clusters[0])
    plt.scatter(x, mean_0, color = colors[0])
    plt.plot(x, mean_0,linewidth=4, color = colors[0])
    mean_1[0] /= len(clusters[1])
    mean_1[1] /= len(clusters[1])
    plt.scatter(x, mean_1, color = colors[1])
    plt.plot(x, mean_1, linewidth=4, color = colors[1])
    mean_2[0] /= len(clusters[2])
    mean_2[1] /= len(clusters[2])
    plt.scatter(x, mean_2, color = colors[2])
    plt.plot(x, mean_2, linewidth=4, color = colors[2])
    plt.ylabel(param)
    plt.show()
    return 


colors    = ['green', 'blue', 'red']

###Clusters à modifier en fonction de l'analyse voulue !!! 
cluster_0 = ['20', '26', '27', '39', '40', '50', '53']
cluster_1 = ['19', '31', '32', '33', '36', '41', '43', '52']
cluster_2 = ['08', '09', '11', '12', '13', '14', '15', '18', '21', '22', '24', '28', '34', '37', '42', '45', '46', '48', '51']
clusters  = [cluster_0, cluster_1, cluster_2]

data = dataframe_mvt(dic1, dic2) 
print(data) 
graph(data, 'qc_mot_abs', patients, clusters, colors)
graph(data, 'qc_cnr_avg', patients, clusters, colors)
graph(data, 'qc_cnr_std', patients, clusters, colors)
graph(data, 'qc_mot_rel', patients, clusters, colors)