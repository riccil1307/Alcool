import os
import shutil

"""
Code pour déplacer des fichiers (se trouvant dans folder_in) 
                   dans un dossier de sortie (folder_out) 
                   
Ce dossier de sortie possède les sous-fichiers suivants : 
   dti_FA, dti_MD, dti_AD, dti_RD
   noddi_fintra, noddi_fextra, noddi_fiso, noddi_odi
   diamond_wFA, diamond_wMD, diamond_wAD, diamond_wRD, diamond_fractions_ftot, diamond_fractions_csf
   mf_CSD_frac_csf, mf_CSD_fvf_tot
   
"""

folder_in  = r'/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/Nifti/'
folder_out = r'/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/Nifti_final/'
  
for file in os.listdir(folder_in) : 
        #DTI 
    if "dti_FA" in file : 
        source = folder_in + file
        destination = folder_out + "dti_FA"
        shutil.move(source, destination)
    if "dti_MD" in file : 
        source = folder_in + file
        destination = folder_out + "dti_MD"
        shutil.move(source, destination)
    if "dti_AD" in file : 
        source = folder_in + file
        destination = folder_out + "dti_AD"
        shutil.move(source, destination)
    if "dti_RD" in file : 
        source = folder_in + file
        destination = folder_out + "dti_RD"
        shutil.move(source, destination)
        #NODDI
    if "noddi_fintra" in file : 
        source = folder_in + file
        destination = folder_out + "noddi_fintra"
        shutil.move(source, destination)
    if "noddi_fextra" in file : 
        source = folder_in + file
        destination = folder_out + "noddi_fextra"
        shutil.move(source, destination)
    if "noddi_fiso" in file : 
        source = folder_in + file
        destination = folder_out + "noddi_fiso"
        shutil.move(source, destination)
    if "noddi_odi" in file : 
        source = folder_in + file
        destination = folder_out + "noddi_odi"
        shutil.move(source, destination)
        #DIAMOND
    if "diamond_wFA" in file : 
        source = folder_in + file
        destination = folder_out + "diamond_wFA"
        shutil.move(source, destination)
    if "diamond_wMD" in file : 
        source = folder_in + file
        destination = folder_out + "diamond_wMD"
        shutil.move(source, destination)
    if "diamond_wAD" in file : 
        source = folder_in + file
        destination = folder_out + "diamond_wAD"
        shutil.move(source, destination)
    if "diamond_wRD" in file : 
        source = folder_in + file
        destination = folder_out + "diamond_wRD"
        shutil.move(source, destination)
    if "diamond_fractions_ftot" in file : 
        source = folder_in + file
        destination = folder_out + "diamond_fractions_ftot"
        shutil.move(source, destination)
    if "diamond_fractions_csf" in file : 
        source = folder_in + file
        destination = folder_out + "diamond_fractions_csf"
        shutil.move(source, destination)
        #MF 
    if "mf_CSD_frac_csf" in file : 
        source = folder_in + file
        destination = folder_out + "mf_CSD_frac_csf"
        shutil.move(source, destination)
    if "mf_CSD_fvf_tot" in file : 
        source = folder_in + file
        destination = folder_out + "mf_CSD_fvf_tot"
        shutil.move(source, destination)