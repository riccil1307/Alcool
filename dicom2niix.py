import os
import pathlib
import subprocess

"""
Code pour convertir des fichiers dicom (se trouvant dans dicom_base_folder) 
                    en fichiers nifti (se trouvant dans niftii_base_folder) 
"""

dicom_base_folder = r"C:\Users\dricotl\Desktop\MELISSA"
niftii_base_folder = r"C:\Users\dricotl\Desktop\LAURA\niftii"

for patient in os.listdir(dicom_base_folder):
    for modality in os.listdir(os.path.join(dicom_base_folder,patient)):

        input_dir = os.path.join(dicom_base_folder, patient, modality)
        if not os.path.isdir(input_dir):
            continue

        if modality[6:] in ["Anat3D", "DTI", "DTIcorr"]:
            modality = modality[6:]
        elif modality[2:] in ["Anat3D", "DTI", "DTIcorr"]:
            modality = modality[2:]
        else:
            continue

        if modality in ["Anat3D", "Restingstate"]:
            output_dir = os.path.join(niftii_base_folder, modality)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            bashCMD = r"D:\Programmes\MRIcron\Resources\dcm2niix.exe -b y " \
                      r"-f sub" + patient[1:6] + "_T1 -z y -o " + output_dir + " " \
                      r" " + input_dir
        else:
            output_dir = os.path.join(niftii_base_folder, modality, patient[4:6])
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            bashCMD = r"D:\Programmes\MRIcron\Resources\dcm2niix.exe -b y " \
                      r"-f sub" + patient[1:6] + " -z y -o " + output_dir + " " \
                      r" " + input_dir
        print(bashCMD)
        process = subprocess.Popen(bashCMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()