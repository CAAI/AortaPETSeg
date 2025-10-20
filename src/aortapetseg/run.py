import os
from subprocess import check_output
import tempfile
from shutil import copy
import torch
from nnunetv2.paths import nnUNet_results
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

def _download_weights():
  """ Download model weights from Zenodo
    https://doi.org/10.5281/zenodo.17357233
  """
  url = "https://zenodo.org/records/17357233/files/weights.zip?download=1"

  if not "Dataset300_aorta_seg" in os.listdir(nnUNet_results):
    if any("Dataset300" in x for x in os.listdir(nnUNet_results)):
      raise Exception("Task with same ID (300) already exists but is not aorta_seg")
    
    print("Downloading weights...")
    check_output(["curl", url, "|", "bsdtar", "-xvf-", "-C", nnUNet_results],shell=True)
    print(f"  Done")

def main(suv_file_path,seg_out_file_path):
  global nnUNet_results

  if nnUNet_results is None:
    raw_dir = os.environ.get("nnUNet_raw_data_base",None)
    assert raw_dir is not None
    nnUNet_results = os.path.join(raw_dir, "nnUNet_results")
  os.makedirs(nnUNet_results,exist_ok=True)
  
  # Download weights if needed
  _download_weights()

  # Configuration    
  dataset = 300
  configuration = '3d_fullres'
  trainer = 'nnUNetTrainer'
  plans = 'nnUNetResEncUNetMPlans'
  folds = (0,1,2,3,4)
    
  # Instantiate the nnUNetPredictor
  predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda'),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
  )
  
  # Initializes the network architecture, loads the checkpoint
  predictor.initialize_from_trained_model_folder(
    os.path.join(nnUNet_results, convert_id_to_dataset_name(dataset), f'{trainer}__{plans}__{configuration}'),
    use_folds=folds
  )
  
  with tempfile.TemporaryDirectory() as in_dir:
    with tempfile.TemporaryDirectory() as out_dir:
      # Copy input file to temp
      copy(suv_file_path,os.path.join(in_dir, "in_0000.nii.gz"))
      
      # Run inference
      predictor.predict_from_files(
        [os.path.join(in_dir, "in_0000.nii.gz")], [os.path.join(out_dir, "segmentation.nii.gz")],
        save_probabilities=False, overwrite=False,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
      )

      # check_output(["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", "300", "-f", "0", "1", "2", "3", "4", "-p", "nnUNetResEncUNetMPlans", "-c", "3d_fullres"])
    
      # out_file = [os.path.join(out_dir, x) for x in os.listdir(out_dir) if str(x).endswith(".nii.gz")]
      # assert len(out_file) == 1
      # out_file = out_file[0]
      # copy(out_file,seg_out_file_path)
      copy(os.path.join(out_dir, "segmentation.nii.gz"),seg_out_file_path)

if __name__ == "__main__":
  import sys
  main(sys.argv[1],sys.argv[2])