#!/usr/bin/env python3
import sys
import os
import subprocess
import threading
from pathlib import Path
PROJECT_ROOT =Path (__file__ ).resolve ().parent
TRAINING_SCRIPT =PROJECT_ROOT /"main.py"
CHECKPOINTS_BASE_DIR =PROJECT_ROOT /"checkpoints"
LOG_OUTPUT_DIR =PROJECT_ROOT /"training_output"
TRAINING_JOBS =[
{
"name":"chk_shuffle_yes_episodic",
"shuffle":"yes",
"strategy":"episodic",
},
{
"name":"chk_shuffle_yes_retrieval",
"shuffle":"yes",
"strategy":"retrieval",
},
{
"name":"chk_shuffle_no_episodic",
"shuffle":"no",
"strategy":"episodic",
},
{
"name":"chk_shuffle_no_retrieval",
"shuffle":"no",
"strategy":"retrieval",
}
]
def run_training_job (job_config :dict ):
    job_name =job_config ["name"]
    shuffle =job_config ["shuffle"]
    strategy =job_config ["strategy"]
    checkpoint_dir =CHECKPOINTS_BASE_DIR /job_name
    log_file_path =LOG_OUTPUT_DIR /f"{job_name }.txt"
    checkpoint_dir .mkdir (exist_ok =True ,parents =True )
    cmd =[
    sys .executable ,
    str (TRAINING_SCRIPT ),
    "--epochs","50",
    "--episodic_label_shuffle",shuffle ,
    "--training_strategy",strategy ,
    "--checkpoint_dir",str (checkpoint_dir ),
    ]
    cmd_str =' '.join (cmd )
    print (f"[Thread: {job_name }] 🚀 STARTING")
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
        if result .returncode ==0 :
            print (f"[Thread: {job_name }] ✅ FINISHED (Success)")
        else :
            print (f"[Thread: {job_name }] ❌ FINISHED (Error: {result .returncode })")
    except Exception as e :
        error_msg =f"--- LAUNCH FAILED ---\nFailed to start {job_name }: {e }\nCommand: {cmd_str }\n"
        print (f"[Thread: {job_name }] 💥 CRITICAL FAILURE: {e }")
        with open (log_file_path ,"a",encoding ="utf-8")as log_file :
            log_file .write (error_msg )
def main ():
    print ("--- 🏁 Starting Batch Training ---")
    if not TRAINING_SCRIPT .exists ():
        print (f"❌ ERROR: main.py not found at {TRAINING_SCRIPT }")
        print ("Please make sure this script is in the same directory.")
        sys .exit (1 )
    LOG_OUTPUT_DIR .mkdir (exist_ok =True )
    print (f"Logs will be saved to: {LOG_OUTPUT_DIR .resolve ()}")
    print (f"Checkpoints will be in: {CHECKPOINTS_BASE_DIR .resolve ()}")
    print ("-"*30 +"\n")
    threads =[]
    for job in TRAINING_JOBS :
        t =threading .Thread (target =run_training_job ,args =(job ,))
        threads .append (t )
        t .start ()
    for t in threads :
        t .join ()
    print ("\n"+"-"*30 )
    print ("🎉 All training jobs are complete.")
    print ("Check log files for details.")
if __name__ =="__main__":
    main ()
