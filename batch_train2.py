import sys
import subprocess
import threading
from pathlib import Path
PROJECT_ROOT =Path (__file__ ).resolve ().parent
TRAINING_SCRIPT =PROJECT_ROOT /"main.py"
CHECKPOINTS_BASE_DIR =PROJECT_ROOT /"checkpoints"
LOG_OUTPUT_DIR =PROJECT_ROOT /"training_output"
BASE_CONFIG ={
"num_experts":4 ,
"embedding_strategy":"learned",
"d_model":256 ,
"n_heads":8 ,
"n_layers":6 ,
}
TRAINING_JOBS =[]
job_moe_2 =BASE_CONFIG .copy ()
job_moe_2 ["name"]="chk_episodic_moe_2"
job_moe_2 ["num_experts"]=2
TRAINING_JOBS .append (job_moe_2 )
job_moe_8 =BASE_CONFIG .copy ()
job_moe_8 ["name"]="chk_episodic_moe_8"
job_moe_8 ["num_experts"]=8
TRAINING_JOBS .append (job_moe_8 )
job_pretrained =BASE_CONFIG .copy ()
job_pretrained ["name"]="chk_episodic_pretrained"
job_pretrained ["embedding_strategy"]="pretrained"
TRAINING_JOBS .append (job_pretrained )
job_d128 =BASE_CONFIG .copy ()
job_d128 ["name"]="chk_episodic_d128"
job_d128 ["d_model"]=128
TRAINING_JOBS .append (job_d128 )
job_d512 =BASE_CONFIG .copy ()
job_d512 ["name"]="chk_episodic_d512"
job_d512 ["d_model"]=512
TRAINING_JOBS .append (job_d512 )
job_h4 =BASE_CONFIG .copy ()
job_h4 ["name"]="chk_episodic_h4"
job_h4 ["n_heads"]=4
TRAINING_JOBS .append (job_h4 )
job_h16 =BASE_CONFIG .copy ()
job_h16 ["name"]="chk_episodic_h16"
job_h16 ["n_heads"]=16
TRAINING_JOBS .append (job_h16 )
job_l4 =BASE_CONFIG .copy ()
job_l4 ["name"]="chk_episodic_l4"
job_l4 ["n_layers"]=4
TRAINING_JOBS .append (job_l4 )
job_l8 =BASE_CONFIG .copy ()
job_l8 ["name"]="chk_episodic_l8"
job_l8 ["n_layers"]=8
TRAINING_JOBS .append (job_l8 )
def run_training_job (job_config :dict ):
    job_name =job_config ["name"]
    checkpoint_dir =CHECKPOINTS_BASE_DIR /job_name
    log_file_path =LOG_OUTPUT_DIR /f"{job_name }.txt"
    checkpoint_dir .mkdir (exist_ok =True ,parents =True )
    cmd =[
    sys .executable ,
    str (TRAINING_SCRIPT ),
    "--epochs","50",
    "--episodic_label_shuffle","yes",
    "--training_strategy","episodic",
    "--checkpoint_dir",str (checkpoint_dir ),
    "--resume",
    ]
    for key ,value in job_config .items ():
        if key =="name":
            continue
        cmd .append (f"--{key }")
        cmd .append (str (value ))
    cmd_str =' '.join (cmd )
    print (f"[Thread: {job_name }] STARTING")
    print (f"[Thread: {job_name }]   > Log: {log_file_path .name }")
    print (f"[Thread: {job_name }]   > Cmd: {cmd_str }\n")
    try :
        with open (log_file_path ,"w",encoding ="utf-8")as log_file :
            result =subprocess .run (
            cmd ,
            stdout =log_file ,
            stderr =subprocess .STDOUT ,
            text =True ,
            check =False
            )
        status ="FINISHED (Success)"if result .returncode ==0 else f"FINISHED (Error {result .returncode })"
        print (f"[Thread: {job_name }] {status }")
    except Exception as e :
        print (f"[Thread: {job_name }] CRITICAL FAILURE: {e }")
        with open (log_file_path ,"a",encoding ="utf-8")as f :
            f .write (f"\n--- LAUNCH FAILED ---\n{e }\n")
def main ():
    print ("--- Starting Full Ablation Batch (MoE=2 + MoE=8 + all previous runs) ---")
    if not TRAINING_SCRIPT .exists ():
        print (f"ERROR: main.py not found at {TRAINING_SCRIPT }")
        sys .exit (1 )
    LOG_OUTPUT_DIR .mkdir (exist_ok =True )
    print (f"Logs → {LOG_OUTPUT_DIR .resolve ()}")
    print (f"Checkpoints → {CHECKPOINTS_BASE_DIR .resolve ()}")
    print (f"Total jobs: {len (TRAINING_JOBS )}\n")
    threads =[]
    for job in TRAINING_JOBS :
        t =threading .Thread (target =run_training_job ,args =(job ,))
        threads .append (t )
        t .start ()
    for t in threads :
        t .join ()
    print ("\nAll jobs completed!")
    print ("To resume any run, just add --resume to the command or launch manually with:")
    print ("  python main.py --resume --checkpoint_dir checkpoints/<folder_name> ...")
if __name__ =="__main__":
    main ()
