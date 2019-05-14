# coding:utf-8
import os
import numpy as np
import SimpleITK as sitk

def read_mhd_and_raw(path, numpyFlag=True):
    """
    This function use sitk
    path : Meta data path
    ex. /hogehoge.mhd
    numpyFlag : Return numpyArray or sitkArray
    return : numpyArray(numpyFlag=True)
    Note ex.3D :numpyArray axis=[z,y,x], sitkArray axis=(z,y,x)
    """
    img = sitk.ReadImage(path)
    if not numpyFlag:
        return img

    nda = sitk.GetArrayFromImage(img)  # (img(x,y,z)->numpyArray(z,y,x))
    return nda


def write_mhd_and_raw(Data, path):
    """
    This function use sitk
    Data : sitkArray
    path : Meta data path
    ex. /hogehoge.mhd
    """
    if not isinstance(Data, sitk.SimpleITK.Image):
        print('Please check your ''Data'' class')
        return False

    data_dir, file_name = os.path.split(path)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    sitk.WriteImage(Data, path, True)

    return True


# load data with list path
def load_matrix_data(path, dtype):
    # load data_list
    data_list = []

    with open(path) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            data_list.append(line[:])

    data = []
    for i in data_list:
        print('image from: {}'.format(i[0]))

        image = read_mhd_and_raw(i[0]).astype(dtype)
        data.append(image)

    data = np.asarray(data)
    return data

# load data with list
def load_data_with_list(list, dtype):
    data = []
    for i in list:
        print('image from: {}'.format(i[0]))

        image = read_mhd_and_raw(i[0]).astype(dtype)
        data.append(image)

    data = np.asarray(data)
    return data

# load list
def load_list(path):
    data_list = []
    with open(path) as paths_file:
        for line in paths_file:
            if not line: continue
            line = line.replace('\n','')
            data_list.append(line[:])
    return data_list

def write_raw(Data, path):
    """
    data : save data as 1D numpy array
    path :  directories + save_name
    ex. /hoge.raw
    """
    data_dir, file_name = os.path.split(path)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    # Data = Data.astype(self.type)
    if Data.ndim != 1:
        print("Please check Array dimensions")
        return False
    with open(path, 'wb') as fid:
        fid.write(Data)

    # Origin code
    # fid = open(path, 'wb')
    # fid.write(Data)
    # fid.close()
    return True