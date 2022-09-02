import elikopy 
import os
import json

"""
Code pour lire les fichiers eddy_corr des patients afin d'avoir accès aux variables :
    Movement_abs, Movement_rel, CNR_avg, CNR_std
qui sont stockées dans deux dictionnaires (dic1, dic2)
"""

#elikopy.utils.get_patient_list_by_types('/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/alcoholic_study_v2/', type=1)
#elikopy.utils.get_patient_list_by_types('/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/alcoholic_study_v2/', type=2)
path1   = '/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/alcoholic_study_v2/subjects/'
path2   = '/dMRI/preproc/eddy/'
path3   = '_eddy_corr.qc/qc.json'

dic1 = {}
dic2 = {}

list_subE1 = ['sub41_E1', 'sub27_E1', 'sub32_E1', 'sub09_E1', 'sub45_E1', 'sub48_E1', 'sub21_E1', 'sub05_E1', 'sub15_E1', 'sub11_E1', 'sub46_E1', 'sub31_E1', 'sub16_E1', 'sub26_E1', 'sub50_E1', 'sub22_E1', 'sub47_E1', 'sub44_E1', 'sub02_E1', 'sub53_E1', 'sub18_E1', 'sub01_E1', 'sub34_E1', 'sub35_E1', 'sub51_E1', 'sub23_E1', 'sub04_E1', 'sub38_E1', 'sub19_E1', 'sub28_E1', 'sub24_E1', 'sub33_E1', 'sub37_E1', 'sub29_E1', 'sub14_E1', 'sub13_E1', 'sub36_E1', 'sub12_E1', 'sub06_E1', 'sub43_E1', 'sub40_E1', 'sub20_E1', 'sub39_E1', 'sub17_E1', 'sub52_E1', 'sub42_E1', 'sub08_E1', 'sub10_E1', 'sub30_E1']
list_subE2 = ['sub32_E2', 'sub29_E2', 'sub53_E2', 'sub31_E2', 'sub39_E2', 'sub07_E2', 'sub45_E2', 'sub13_E2', 'sub24_E2', 'sub42_E2', 'sub11_E2', 'sub25_E2', 'sub21_E2', 'sub08_E2', 'sub01_E2', 'sub09_E2', 'sub50_E2', 'sub22_E2', 'sub34_E2', 'sub30_E2', 'sub37_E2', 'sub12_E2', 'sub20_E2', 'sub17_E2', 'sub14_E2', 'sub04_E2', 'sub05_E2', 'sub06_E2', 'sub33_E2', 'sub35_E2', 'sub43_E2', 'sub27_E2', 'sub28_E2', 'sub19_E2', 'sub15_E2', 'sub26_E2', 'sub02_E2', 'sub40_E2', 'sub18_E2', 'sub52_E2', 'sub51_E2', 'sub36_E2', 'sub48_E2', 'sub46_E2', 'sub03_E2', 'sub41_E2']

for pE1 in list_subE1:
    full_path = path1 + pE1 + path2 + pE1 + path3
    metrics = {}
    with open(full_path) as f:
        data = json.load(f)
        dic1[pE1] = metrics
        metrics['Patients'] = pE1[3:5]
        metrics['Time'] = pE1[7:8]
        metrics['Movement_abs'] = data['qc_mot_abs']
        metrics['Movement_rel'] = data['qc_mot_rel']
        metrics['CNR_avg'] = data['qc_cnr_avg'][0]
        metrics['CNR_std'] = data['qc_cnr_std'][0]
        
for pE2 in list_subE2:
    full_path = path1 + pE2 + path2 + pE2 + path3
    metrics2 ={}
    with open(full_path) as f:
        data = json.load(f)
        dic2[pE2] = metrics2
        metrics2['Patients'] = pE2[3:5]
        metrics2['Time'] = pE2[7:8]
        metrics2['Movement_abs'] = data['qc_mot_abs']
        metrics2['Movement_rel'] = data['qc_mot_rel']
        metrics2['CNR_avg'] = data['qc_cnr_avg'][0]
        metrics2['CNR_std'] = data['qc_cnr_std'][0]


with open("dic1.json", "w") as i :
   json.dump(dic1, i)
with open("dic2.json", "w") as i :
   json.dump(dic2, i)