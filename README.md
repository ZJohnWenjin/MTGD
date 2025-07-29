# MTGD
MULTI-SEQUENCE MRI TO MULTI-TRACER PET GENERATION VIA DIFFUSION MODEL

## 🔧 Configuration

Modify the configuration file at [`configs/myconfig.yaml`](configs/myconfig.yaml) to customize your MRI inputs and PET targets.

```yaml
pet: 
  # PET types as target
  pet_modalities: ['pet_type1.nii.gz', 'pet_type2.nii.gz']

mri:
  # MRI sequences as input
  mri_sequence: ['mri_type1.nii.gz', 'mri_type2.nii.gz', 'mri_type3.nii.gz']

folder_path:
  path: "PATH/TO/DATA"

```
Each patient's folder must include all corresponding .nii.gz files.
> 📌 Make sure that the file names match those specified in `mri_sequence` and `pet_modalities` in your config file.
```
PATH/TO/DATA/
├── patient1/
│   ├── mri_type1.nii.gz
│   ├── mri_type2.nii.gz
│   ├── pet_type1.nii.gz
│   └── ...
├── patient2/
│   ├── mri_type1.nii.gz
│   ├── mri_type2.nii.gz
│   └── ...

