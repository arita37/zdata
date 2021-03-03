import subprocess
def run_cafem():
    p = subprocess.Popen(['python','cafem.py','--out_dir','out/ml','--num_epochs','1','--meta_batch_size','1','--num_episodes','1'])
    return p

def run_single_afem():
    p = subprocess.Popen(['python','single_afem.py','--load_weight','out/ml/cafem/model_1.ckpt','--dataset','1049','--out_dir','out/o2_5_1049','--num_epochs','1','--buffer_size','1000','--num_episodes','1'])
    return p
