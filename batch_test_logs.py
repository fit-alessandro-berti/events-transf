import re
import argparse
import subprocess
import sys
from pathlib import Path
def alphanumeric (s :str )->str :
    return re .sub (r"[^a-zA-Z0-9_]","",s .replace (" ","_"))
def main ():
    parser =argparse .ArgumentParser (
    description ="Batch-run testing.py on logs starting with >=3 zeros. "
    "Safe to resume: already processed files are skipped."
    )
    parser .add_argument ("--checkpoint_dir",type =str ,required =True ,
    help ="Checkpoint directory (used for model loading AND output subfolder)")
    parser .add_argument ("--checkpoint_epoch",type =int ,default =None ,
    help ="Specific epoch to test (default: latest). "
    "If provided, epoch is added to the output subfolder name.")
    parser .add_argument ("--test_log_name",type =str ,default =None ,
    help ="Default test log name (overridden per file anyway)")
    parser .add_argument ("--test_mode",type =str ,default =None ,
    choices =["meta_learning","retrieval_augmented"],
    help ="Force a specific test mode")
    parser .add_argument ("--test_retrieval_prediction_mode",type =str ,default =None ,
    choices =["proto_head","foundation_knn"],
    help ="Retrieval prediction mode when --test_mode retrieval_augmented is used.")
    parser .add_argument ("--test_retrieval_report_confidence_buckets",action ="store_true",
    help ="Enable 5 confidence-bucket metrics in retrieval evaluation (proto_head mode).")
    args =parser .parse_args ()
    project_root =Path (__file__ ).resolve ().parent
    logs_dir =project_root /"logs"
    testing_script =project_root /"testing.py"
    base_output_dir =project_root /"testing_output"
    base_output_dir .mkdir (exist_ok =True )
    checkpoint_name =Path (args .checkpoint_dir ).name
    chk_part =alphanumeric (checkpoint_name )
    if args .checkpoint_epoch is not None :
        subfolder_name =f"{chk_part }_{args .checkpoint_epoch }"
        epoch_part =str (args .checkpoint_epoch )
    else :
        subfolder_name =chk_part
        epoch_part ="latest"
    checkpoint_output_dir =base_output_dir /subfolder_name
    checkpoint_output_dir .mkdir (exist_ok =True )
    if not testing_script .exists ():
        print (f"ERROR: testing.py not found at {testing_script }")
        sys .exit (1 )
    pattern =re .compile (r"^0{3,}.*\.xes(\.gz)?$",re .IGNORECASE )
    all_candidate_logs =sorted (
    p for p in logs_dir .iterdir ()if p .is_file ()and pattern .match (p .name )
    )
    if not all_candidate_logs :
        print ("No logs found starting with 000 or more zeros.")
        return
    mode_part =alphanumeric (args .test_mode )if args .test_mode else "defaultmode"
    cmd_base =[
    sys .executable ,str (testing_script ),
    "--checkpoint_dir",str (args .checkpoint_dir ),
    ]
    if args .checkpoint_epoch is not None :
        cmd_base +=["--checkpoint_epoch",str (args .checkpoint_epoch )]
    if args .test_log_name :
        cmd_base +=["--test_log_name",args .test_log_name ]
    if args .test_mode :
        cmd_base +=["--test_mode",args .test_mode ]
    if args .test_retrieval_prediction_mode :
        cmd_base +=["--test_retrieval_prediction_mode",args .test_retrieval_prediction_mode ]
    if args .test_retrieval_report_confidence_buckets :
        cmd_base +=["--test_retrieval_report_confidence_buckets"]
    processed =skipped =0
    print (f"Found {len (all_candidate_logs )} candidate logs")
    print (f"Output directory → {checkpoint_output_dir }\n")
    for log_path in all_candidate_logs :
        log_stem =re .sub (r'\.xes(\.gz)?$','',log_path .name ,flags =re .IGNORECASE )
        log_part =alphanumeric (log_stem )
        out_filename =f"{chk_part }_{log_part }_{mode_part }_{epoch_part }.txt"
        out_path =checkpoint_output_dir /out_filename
        if out_path .exists ():
            print (f"✓ SKIP (exists): {log_path .name }")
            skipped +=1
            continue
        print (f"→ Testing: {log_path .name } → {out_path .name }")
        cmd =cmd_base .copy ()
        cmd +=["--test_log_name",str (log_path )]
        try :
            result =subprocess .run (
            cmd ,
            stdout =subprocess .PIPE ,
            stderr =subprocess .STDOUT ,
            text =True ,
            check =False ,
            timeout =3600 ,
            )
            with open (out_path ,"w",encoding ="utf-8")as f :
                f .write (f"COMMAND: {' '.join (cmd )}\n")
                f .write (f"RETURN CODE: {result .returncode }\n")
                f .write ("="*80 +"\n")
                f .write (result .stdout )
            status ="OK"if result .returncode ==0 else f"ERR {result .returncode }"
            print (f"   → Finished [{status }]")
            processed +=1
        except subprocess .TimeoutExpired :
            msg =f"TIMEOUT after 1h: {log_path .name }\n"
            print (msg .strip ())
            with open (out_path ,"w",encoding ="utf-8")as f :
                f .write (msg )
            processed +=1
        except Exception as e :
            msg =f"FAILED: {log_path .name } → {e }\n"
            print (msg .strip ())
            with open (out_path ,"w",encoding ="utf-8")as f :
                f .write (msg )
            processed +=1
    print ("\n"+"="*60 )
    print (f"DONE! Processed: {processed } | Skipped: {skipped }")
    print (f"All results → {checkpoint_output_dir }")
    print ("="*60 )
if __name__ =="__main__":
    main ()
