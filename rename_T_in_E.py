import os
from perso_path import perso_path_string, patient_number_list

"""
Code pour changer le nom de certains fichiers : 
    T1 et T2 deviennent E1 et E2 
"""

patient_number = ["02","04","05","08","09","11","12","13","14","15","17","18","19","20","21","22","24","26","27","28","30","31","32","33","34","35","36","37","39","40","41","42","43","45","46"]
time_list      = ["T1", "T2"]
new_time_list  = ["E1", "E2"]

for patient_nb in patient_number:
    for time, new_time in zip(time_list, new_time_list):
        nouveau_folder = "/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/alcoholic_study/subjects/sub" + patient_nb + "_" + new_time + "/T1/"
        list_directory = os.listdir(nouveau_folder)
        
        for i in list_directory:   
            if ("_T1_T1" in i):
                new_i = i.replace("_T1_T1", "_E1_T1")
                ancien = nouveau_folder + i
                nouveau = nouveau_folder + new_i
                os.rename(ancien, nouveau)
            
            if ("T2" in i):
                new_i = i.replace("T2", "E2")
                ancien = nouveau_folder + i
                nouveau = nouveau_folder + new_i
                os.rename(ancien, nouveau)
                