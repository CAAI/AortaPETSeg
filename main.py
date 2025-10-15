import os
import subprocess
import tempfile
import shutil

def main(suv_file_path,seg_out_file_path):
    results_dir = os.environ.get("nnUNet_results",None)
    if results_dir is None:
        raw_dir = os.environ.get("nnUNet_raw_data_base",None)
        assert raw_dir is not None
        results_dir = raw_dir + "/nnUNet_results"
    os.makedirs(results_dir,exist_ok=True)
    if not "Dataset300_aorta_seg" in os.listdir(results_dir):
        if any("Dataset300" in x for x in os.listdir(results_dir)):
            raise Exception("Task with same ID (300) already exists but is not aorta_seg")
        print("Downloading weights...")
        
        #subprocess.check_output(["curl", "https://zenodo.org/records/13691360/files/Dataset214_LUMIERE_default_TL.zip?download=1", "|", "bsdtar", "-xvf-", "-C", results_dir],shell=True)
        subprocess.check_output(["unzip","/homes/ulrich/projects/AortaPETSeg/nnUNet_results/weights.zip", "-d", results_dir])
        
    
    with tempfile.TemporaryDirectory() as in_dir:
        with tempfile.TemporaryDirectory() as out_dir:
            shutil.copy(suv_file_path,in_dir+"/in_0000.nii.gz")
            subprocess.check_output(["nnUNetv2_predict", "-i", str(in_dir), "-o", str(out_dir), "-d", "300", "-f", "0", "1", "2", "3", "4", "-p", "nnUNetResEncUNetMPlans", "-c", "3d_fullres"])
        
            out_file = [os.path.join(out_dir, x) for x in os.listdir(out_dir) if str(x).endswith(".nii.gz")]
            assert len(out_file) == 1
            out_file = out_file[0]
            shutil.copy(out_file,seg_out_file_path)

if __name__ == "__main__":
    import sys
    main(sys.argv[1],sys.argv[2])