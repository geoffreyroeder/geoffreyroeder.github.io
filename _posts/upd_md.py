# modified 2021-3-28 to allow hyphens in blog post names
import fileinput, re
for f in fileinput.input(inplace=True):
    print(re.sub(r'^(!.*]\()(\w+(?:-\w+)+_files/)', r'\1/images/\2', f), end='')
