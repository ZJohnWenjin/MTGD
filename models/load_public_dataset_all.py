
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import random



def split_and_label(final_pet_lists, train_ratio=0.8):
    train_set = []
    test_set = []

    for pet_list in final_pet_lists:
        random.shuffle(pet_list)  # Shuffle the list to ensure randomness
        split_idx = int(len(pet_list) * train_ratio)
        train_set.extend(pet_list[:split_idx])
        test_set.extend(pet_list[split_idx:])

    return train_set, test_set


def generate_nifti_structure(folder_path,structural_mri,pet_images):
    # List all subdirectories in the folder
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    pet_types = {}
    for subfolder in subfolders:
        # List all files in the subfolder
        files = os.listdir(subfolder)

        # Define file categories
        structural_mri = structural_mri
        pet_images = pet_images

        # Initialize lists for structural MRI and PET
        mri_list = [
            os.path.join(subfolder, file) if file in files else ''
            for file in structural_mri
        ]

        for pet in pet_images:
            if pet in files:
                if pet not in pet_types:
                    pet_types[pet] = []
                pet_types[pet].append([mri_list, os.path.join(subfolder, pet), pet_images.index(pet)])

    # Compile the final list of PET type-specific lists
    final_pet_lists = [pet_types[pet] for pet in pet_types if pet_types[pet]]
    train_set, test_set = split_and_label(final_pet_lists)

    return train_set, test_set



def generate_nifti_structure_test(folder_path,structural_mri,pet_images):
    pet_types = {}
    
    # List all files in the subfolder
    files = os.listdir(folder_path)

    # Define file categories
    structural_mri = structural_mri
    pet_images = pet_images

    # Initialize lists for structural MRI and PET
    mri_list = [
        os.path.join(folder_path, file) if file in files else ''
        for file in structural_mri
    ]

    for pet in pet_images:
        if pet in files:
            if pet not in pet_types:
                pet_types[pet] = []
            pet_types[pet].append([mri_list, os.path.join(folder_path, pet), pet_images.index(pet)])

    # Compile the final list of PET type-specific lists
    final_pet_lists = [pet_types[pet] for pet in pet_types if pet_types[pet]]
    train_set, test_set = split_and_label(final_pet_lists,1)

    return train_set

class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        # get pet type
        pet_type = torch.tensor(self.dataset[index][2])

        # get label
        labels_nii = nib.load(self.dataset[index][1])
        Mask_numpy = labels_nii.get_fdata()
        Mask_numpy = self.min_max_normalize(Mask_numpy)
        label = torch.from_numpy(Mask_numpy)

        # get input
        input_numpy_list = []
        for k in range(len(self.dataset[index][0])):
            if len(self.dataset[index][0][k]) == 0:
                input_numpy_list.append(np.zeros_like(Mask_numpy))
            else:
                CT_nii = nib.load(self.dataset[index][0][k])
                CT_numpy = CT_nii.get_fdata()
                CT_numpy = self.min_max_normalize(CT_numpy)
                input_numpy_list.append(CT_numpy)

        input_numpy_list = np.array(input_numpy_list)
        # (1,x,y,z)
        inputs = torch.from_numpy(input_numpy_list)

        return inputs, label, pet_type

    def __len__(self):
        return len(self.dataset)

    def min_max_normalize(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)

        normalized_arr = (arr - min_val) / (max_val - min_val)

        if max_val == 0:
          return np.zeros_like(arr)
        return normalized_arr


required_files = [
    '18F-AV45_ISO_BFC_rigid_normalised_brain.nii.gz',
    '18F-AV1451_ISO_BFC_rigid_normalised_brain.nii.gz'
]
mri = ['T2_ISO_BFC_normalised_brain.nii.gz', 'FLAIR_ISO_BFC_rigid_normalised_brain.nii.gz','T1_ISO_BFC_rigid_normalised_brain.nii.gz']
path = "/g/data/iq24/public_PET/PET_preprosessing/MRI/MRI_dataset/ADNI/021_S_10114"



def load_data(structural_mri,pet_images,folder_path,train=True):

    folder_path = folder_path  # Replace with your folder path
    train_set, test_set = generate_nifti_structure(folder_path,structural_mri,pet_images)

    if train:
        my_dataset = MyDataset(train_set)
    #else:
    #    my_dataset = MyDataset(test_set)
    
    else:
        final_pet_lists = generate_nifti_structure_test(path,mri,required_files)
        my_dataset = MyDataset(final_pet_lists)

    loader = DataLoader(
        my_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False
    )

    return loader
    


def find_folders_with_specific_files(folder_path, required_files):
    # List all subdirectories in the folder
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    matching_folders = []

    for subfolder in subfolders:
        # List all files in the subfolder
        files = os.listdir(subfolder)

        # Check if all required files are in the subfolder
        if all(file in files for file in required_files):
            matching_folders.append(subfolder)

    return matching_folders
#folder_path = "/g/data/iq24/public_PET/PET_preprosessing/MRI/MRI_dataset/ADNI/"
#matching_folders = find_folders_with_specific_files(folder_path, required_files)
#print("Folders with required files:", matching_folders)







