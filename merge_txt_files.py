#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
def merge_txt_files (input_dir :str ,output_file :str ,recursive :bool =True ,encoding :str ="utf-8"):
    input_path =Path (input_dir ).resolve ()
    output_path =Path (output_file ).resolve ()
    if not input_path .is_dir ():
        raise ValueError (f"The input directory '{input_dir }' does not exist or is not a directory.")
    with open (output_path ,"w",encoding =encoding )as outfile :
        pattern ="**/*.txt"if recursive else "*.txt"
        for txt_file in sorted (input_path .glob (pattern )):
            if not txt_file .is_file ():
                continue
            relative_path =txt_file .relative_to (input_path .parent if input_path .parent !=input_path else input_path )
            outfile .write (f"# FILE NAME: {relative_path }\n")
            try :
                with open (txt_file ,"r",encoding =encoding )as infile :
                    for line in infile :
                        outfile .write (line )
                outfile .write ("\n\n")
            except UnicodeDecodeError :
                print (f"Warning: Skipping {txt_file } because it is not valid {encoding }")
            except Exception as e :
                print (f"Error reading {txt_file }: {e }")
    print (f"All .txt files merged into: {output_path }")
def main ():
    parser =argparse .ArgumentParser (
    description ="Merge all .txt files in a directory into one big file, prefixing each with its filename."
    )
    parser .add_argument ("input_dir",help ="Directory containing the .txt files")
    parser .add_argument ("output_file",help ="Path of the merged output file")
    parser .add_argument ("-r","--no-recursive",action ="store_true",help ="Do not search subdirectories")
    parser .add_argument ("-e","--encoding",default ="utf-8",help ="Text encoding (default: utf-8)")
    args =parser .parse_args ()
    merge_txt_files (
    input_dir =args .input_dir ,
    output_file =args .output_file ,
    recursive =not args .no_recursive ,
    encoding =args .encoding
    )
if __name__ =="__main__":
    main ()
