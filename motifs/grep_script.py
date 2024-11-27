import re

def grep_unique_keywords(input_file, keywords_file, output_file):
    # Load keywords from file and create a regular expression pattern
    with open(keywords_file, 'r') as kf:
        keywords = [line.strip() for line in kf if line.strip()]
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b')

    # Track matched keywords to avoid duplicates
    matched_keywords = set()
    matches = []

    with open(input_file, 'r') as infile:
        first_line = infile.readline()
        matches.append(first_line)  # Add the first line to the output
        
        # Search for lines containing any of the keywords
        for line in infile:
            for keyword in keywords:
                if keyword not in matched_keywords and re.search(rf'\b{re.escape(keyword)}\b', line):
                    matches.append(line)
                    matched_keywords.add(keyword)
                    break  # Move to the next line after the first keyword match

    # Write the selected lines to the output file
    with open(output_file, 'w') as outfile:
        outfile.writelines(matches)
# Usage
input_file='/scratch/ssh506/git/dfdl_imputation/motifs/motifs-v9-nr.hgnc-m0.001-o0.0.tbl'
keywords_file = '/scratch/ssh506/git/dfdl_imputation/tfs/allTFs_hg38.txt' # Replace with your keywords file path
output_file = '/scratch/ssh506/git/dfdl_imputation/motifs/motifs-filtered.tbl'     # Replace with your output file path

grep_unique_keywords(input_file, keywords_file, output_file)

