import os
import pandas as pd
import scipy.io as sio
from list_files_with_extension import *

def excel_to_mat(excel_file, sheets_to_convert, mat_file, channel_names):
    # Read data from Excel sheets into pandas DataFrames
    df_dict = {}
    for sheet in sheets_to_convert:
        df_dict[sheet] = pd.read_excel(excel_file, sheet_name=sheet)

    # Create variables in a dictionary
    data_dict = {}
    for channel_name, sheet in zip(channel_names, sheets_to_convert):
        data_dict[channel_name] = df_dict[sheet].values

    # Save the dictionary as a .mat file
    sio.savemat(mat_file, data_dict)
    print('Data saved to {}'.format(mat_file))

def batch_convert_excel_to_mat(excel_folder_path, output_folder_path, sheets_to_convert, channel_names):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    file_paths, file_names = list_files_with_extension(excel_folder_path, extension='.xlsx')

    # for filename in os.listdir(excel_folder_path):
    for  file_path, file_name in zip(file_paths, file_names):
        if file_name.endswith(".xlsx"):
            excel_file_path = os.path.join(excel_folder_path, file_path, file_name)
            os.makedirs(os.path.join(output_folder_path, file_path))
            mat_file_path = os.path.join(output_folder_path, file_path, 'segmentation.mat')
            excel_to_mat(excel_file_path, sheets_to_convert, mat_file_path, channel_names)

if __name__ == '__main__':

    # Example usage:
    # excel_folder_path = 'D:\\0_TRABAJO\\202202\\data\\clf_check\\segmentations_xlsx\\TPEHGDB\\'
    # output_folder_path = 'D:\\0_TRABAJO\\202202\\data\\clf_check\\segmentations_mat\\TPEHGDB\\'
    excel_folder_path = 'F:\\TPL\\segmentation\\TPL_database\\xlsx\\'
    output_folder_path = 'F:\\TPL\\segmentation\\TPL_database\\mat\\'

    sheets_to_convert = ['Channel_4', 'Channel_5', 'Channel_6']
    channel_names = ['Bipolar', 'Bipolar2', 'Bipolar3']
    # sheets_to_convert = ['Channel_1', 'Channel_3', 'Channel_5']


    batch_convert_excel_to_mat(excel_folder_path, output_folder_path, sheets_to_convert, channel_names)
