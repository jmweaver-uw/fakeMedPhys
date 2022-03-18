import pandas as pd

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def write_to_text(pathList):
    f = open('titles.txt', 'w')
    for path in pathList:
        df = pd.read_csv(path)
        titles = df.Title
        for i in range(len(titles)):
            print(titles[i])
            f.write('\n')
            f.write(titles[i])
    f.close()


pathList = ['csv-Medicalphy-set_1974-2011.csv',
            'csv-Medicalphy-set_2013-2022.csv']

write_to_text(pathList)
