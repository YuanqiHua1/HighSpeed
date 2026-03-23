import os

my_folder = r'C:\joetailfit2_Improved\CRISPR_'

file_names = os.listdir(my_folder)
output = ''

for file in file_names:
    filename, file_extension = os.path.splitext(file)
    if file_extension == '.txt':
        filepath = my_folder + '\\' + file
        with open(filepath, 'r') as f:
            content = f.readlines()[0]
            output += content
            break
for file in file_names:
    filename, file_extension = os.path.splitext(file)
    if file_extension == '.txt':
        filepath = my_folder + '\\' + file
        with open(filepath, 'r') as f:
            content = f.readlines()[1]
            output += content + '\n'
with open(my_folder + '\merged_files', 'wb') as merged_files:
   merged_files.write(output)
   merged_files.flush()
   merged_files.close()
