''' dataset-differ

This is a simple script to find the difference between two CSV files
with same column labels. It creates a new CSV file with 2 columns (Old, New)
for each column in the source CSVs.

The new CSV only contains rows that have any difference in the source CSVs,
and then highlight the columns that have changed.

All NaN values will be replaced with '' (Empty string) in the result XLSX file.

You need to provide 2 params:
    
    string - path : The directory in which to look for source csv files.
        Default = '.'
        All files that contain 'csv' in their filenames (including file extention)
        are selected and sorted in alphabetical order.
        Then the last two of them are used as source CSVs.
    
    string - id_col : The column used to uniquely identify each row.
        Default = 'id'

The script will create 'styled.xlsx' in the current directory with the results.

This script was created with the following module version
    numpy==1.16.4
    openpyxl==2.6.2
    pandas==0.24.2
    progressbar==2.5
    xlwt==1.3.0

Date: 28/06/19
Author: V S Sreejeet https://github.com/sreejeet
'''

import pandas as pd
import numpy as np
import os
import sys
from progressbar import ProgressBar, Percentage, Bar


# Cell highlight color
HIGHLIGHT_COLOR = 'yellow'


def get_differ(path:str='.', id_col:str='id'):

    # files
    old = None
    new = None

    files = sorted([file for file in os.listdir(path) if '.csv' in file], key=str.lower)

    old_file = os.path.join(path, files[-2])
    new_file = os.path.join(path, files[-1])

    print('Using files {} and {}'.format(old_file, new_file))
    old = pd.read_csv(old_file)
    new = pd.read_csv(new_file)

    old = old.set_index(id_col)
    new = new.set_index(id_col)

    # Provision for removing columns before processing
    # remove_columns = ['scraped_on']
    # for col in remove_columns:
    #     del old[col]
    #     del new[col]

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(old)).start()
    i = 0
    print('Removing unchanged rows...')
    pbar.start()
    for index in old.index:
        i+=1
        pbar.update(i)
        try:
            if old.loc[index].equals(new.loc[index]):
                old = old.drop(index)
                new = new.drop(index)
        except KeyError:
            continue

    pbar.finish()

    df_all = pd.concat([old, new], axis='columns', keys=['Old', 'New'])
    df_final = df_all.swaplevel(axis='columns')[old.columns[:]]

    # Custom style function
    def highlight_diff(data, color=HIGHLIGHT_COLOR):
        attr = 'background-color: %s' % color
        other = data.xs('Old', axis='columns', level=-1)
        return pd.DataFrame(np.where(data.ne(other, level=0), attr, ''),
            index=data.index, columns=data.columns)

    df_final = df_final.fillna('')
    s = df_final.style.apply(highlight_diff, axis=None)

    # print(old.head())
    # print(new.head())
    # print(df_final.head())

    print('Converting to excel...')
    s.to_excel('styled.xlsx', engine='openpyxl')
    print('Done.')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        get_differ(sys.argv[1], sys.argv[2])
    else:
        get_differ()
