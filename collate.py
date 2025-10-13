import os


def collate_python_files(output_filename="collated_python_files.txt"):
    # Get all .py files in the current directory
    py_files = [f for f in os.listdir('.') if f.endswith('.py') and f != os.path.basename(__file__)]

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for filename in sorted(py_files):
            outfile.write(f"\n{'=' * 80}\n")
            outfile.write(f"# File: {filename}\n")
            outfile.write(f"{'=' * 80}\n\n")

            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

    print(f"✅ Collated {len(py_files)} Python files into '{output_filename}'.")


if __name__ == "__main__":
    collate_python_files()
