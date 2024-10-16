import pandas as pd
import radiomics.featureextractor as featureextractor
import SimpleITK as sitk
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="the task name")
args = parser.parse_args()

settings = {'binWidth': 25,
            'resampledPixelSpacing': [0.8, 0.8, 1],
            'interpolator': sitk.sitkLinear,
            'normalize': True}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableImageTypes(Original={}, LoG={"sigma": [3.0, 4.0, 5.0]}, Wavelet={})

task = args.task
img_path = f"./data/{task}/TCIA-{task}/ct_nii/"
mask_path = f"./data/{task}/TCIA-{task}/mask_nii/"
save_path = f"./data/{task}/TCIA-{task}/radiomics_single_files/"
os.makedirs(save_path, exist_ok=True)

files = sorted(os.listdir(img_path))

for file in tqdm(files):
    imageFile = os.path.join(img_path, file)
    maskFile = os.path.join(mask_path, file.replace("_0000", ""))

    featureVector = extractor.execute(imageFile, maskFile)
    df = pd.DataFrame.from_dict(featureVector.values()).T
    df.columns = featureVector.keys()
    df.insert(0, 'imageFile', imageFile)
    df.to_csv(os.path.join(save_path, f'{file.split(".")[0].replace("_0000", "")}.csv'), index=False)



