import os
from pathlib import Path
from typing import Iterable ,List ,Set
def collate_python_files (
output_filename :str ="collated_python_files.txt",
directories :Iterable [str ]=(".","components"),
exclude_dirs :Iterable [str ]=("__pycache__",".git",".venv","venv",".mypy_cache",".pytest_cache"),
exclude_files :Iterable [str ]=(),
)->None :
    this_file =Path (__file__ ).resolve ()
    roots :List [Path ]=[Path (d ).resolve ()for d in directories ]
    py_files :Set [Path ]=set ()
    for root in roots :
        if not root .exists ():
            continue
        for dirpath ,dirnames ,filenames in os .walk (root ):
            dirnames [:]=[d for d in dirnames if d not in exclude_dirs ]
            for fname in filenames :
                if not fname .endswith (".py"):
                    continue
                if fname .startswith ("batch_"):
                    continue
                if fname in ("collate.py","merge_txt_files.py"):
                    continue
                if fname in exclude_files :
                    continue
                p =Path (dirpath )/fname
                try :
                    if p .resolve ()==this_file :
                        continue
                except FileNotFoundError :
                    continue
                py_files .add (p )
    sorted_files =sorted (py_files ,key =lambda p :(str (p .parent ),p .name ))
    with open (output_filename ,"w",encoding ="utf-8")as outfile :
        for p in sorted_files :
            rel =p if any (str (p ).startswith (str (r ))for r in roots )else p .name
            header_path =str (rel )
            outfile .write (f"\n{'='*80 }\n")
            outfile .write (f"# File: {header_path }\n")
            outfile .write (f"{'='*80 }\n\n")
            try :
                with open (p ,"r",encoding ="utf-8")as infile :
                    outfile .write (infile .read ())
            except UnicodeDecodeError :
                with open (p ,"r",encoding ="latin-1",errors ="replace")as infile :
                    outfile .write (infile .read ())
            outfile .write ("\n\n")
    print (f"✅ Collated {len (sorted_files )} Python files from {len (roots )} dirs into '{output_filename }'.")
if __name__ =="__main__":
    collate_python_files (directories =(".","components"),exclude_dirs =("util",))
