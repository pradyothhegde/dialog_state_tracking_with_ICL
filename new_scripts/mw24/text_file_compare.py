# The program compares two text files line by line and checks if the strings in each line are identical.

# python text_file_compare.py /mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4/processed_data/MW24_OP_ST_PH-nm_SU1_Labse_NN-10_U_SKV_back/MW24_OP_ST_PH-nm_SU1_Labse_NN-10_U_SKV.txt /mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4/processed_data/MW24_OP_ST_PH-nm_SU1_Labse_NN-10_U_SKV/MW24_OP_ST_PH-nm_SU1_Labse_NN-10_U_SKV.txt

import sys

def compare_files_identical(file1_path, file2_path):
    """
    Compares two files line by line and checks if the *strings* in each line are identical.
     Returns True if the files are identical line by line, False otherwise.

    Args:
        file1_path (str): The path to the first file.
        file2_path (str): The path to the second file.
    """
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            while True:
                line1 = file1.readline()
                line2 = file2.readline()

                # If both are empty, end of both files reached simultaneously
                if not line1 and not line2:
                    return True  # Files are identical

                # If one is empty and other is not it means files have different number of lines
                if not line1 or not line2:
                   return False # Files are not identical

                # Remove trailing newline characters for accurate string comparison
                line1_str = line1.rstrip('\n')
                line2_str = line2.rstrip('\n')
                
                if line1_str != line2_str:
                    return False # Files are not identical

    except FileNotFoundError:
        print("Error: One or both files not found.")
        return False # Files are not identical
    except Exception as e:
        print(f"An error occurred: {e}")
        return False # Files are not identical

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <file1_path> <file2_path>")
    else:
        file1_path = sys.argv[1]
        file2_path = sys.argv[2]
        if compare_files_identical(file1_path, file2_path):
           print("Files are identical.")
        else:
           print("Files are not identical.")


# Write the program description here:
