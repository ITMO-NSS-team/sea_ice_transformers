import numpy as np
from torch.utils.data import Dataset
import os
import torch
from typing import List, Union,Tuple
from datetime import datetime, timedelta
from skimage.transform import resize


class IceDataset(Dataset):
    """Dataset for time siries ice mask
    """

    def __init__(self, list_dir: List[str],
                 path_to_dir: str,
                 in_period: int = 0,
                 predict_period: int = 0,
                 stride: int = 1,
                 test_end: Union[int, None] = None,
                 from_ymd: Union[list, None] = None,
                 to_ymd: Union[list, None] = None,
                 pad:bool = False,
                 shuffle=True,
                 resize_img: Union[List[int], None] = None,
                 shift: int =1
                 ):
        """_summary_

        Args:
            list_dir (List[str]): Name of paths with .npy ice matrix
            in_period (int, optional): period of feature sequence. Defaults to 0.
            predict_period (int, optional): period of predicted sequence. Defaults to 0.
            stride (int, optional): Step to split data (If need to split data by week, month or other), defined by day. Defaults to 1.
            test_end (List[int,None], optional): Value to split data,
              if needed to use part of data in 'path_to_dir'[:test_end] directory. Defaults to None.
            from_ymd List[str] : [Year,month,day] from date that you want to choose data (included this date)
            to_ymd List[str] : [Year,month,day] to date that you want to choose data (included this date)
        """
        self.shuffle=shuffle
        self.path_to_dir = path_to_dir
        self.list_dir = list_dir
        self.period = in_period
        self.predict_period = predict_period
        self.stride = stride
        self.shift = shift
        self.data = []
        self.dates = []
        self.end = test_end
        self.from_ymd = from_ymd
        self.to_ymd = to_ymd
        self.pad = pad
        self.resize = resize_img
        self.__load_npz__()

    def __len__(self):
        self.lent = [i for i in range(self.data.shape[1]-self.period-self.predict_period+1)]
        return int((self.data.shape[1]-self.period+1)/self.shift)#-self.predict_period
    def true_size(self):
        return self.true_size

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            list: Of 0- train data. 1 - test data. 2 - train dates. 3 -test dates.
        """
        # if self.shuffle:
        #     idx = random.choice(self.lent)
        #     self.lent.remove(idx)
        if self.pad:
            self.pad_shape= (self.data[:,idx*self.shift:idx*self.shift+self.period].shape[0],
                             self.data[:,idx*self.shift:idx*self.shift+self.period].shape[1],
                             np.max(self.data[:,idx*self.shift:idx*self.shift+self.period].shape),
                             np.max(self.data[:,idx*self.shift:idx*self.shift+self.period].shape))
            
            return np.pad(self.data[:,idx:idx+self.period],
                           [(0,i) for i in np.subtract(self.pad_shape, self.data[:,idx:idx+self.period].shape)],
                             'constant',
                               constant_values=0), self.data[:,idx+self.period:idx+self.period+self.predict_period], self.dates[idx:idx+self.period], self.dates[idx+self.period:idx+self.period+self.predict_period]
        else:
            return self.data[:,idx*self.shift:idx*self.shift+self.period], self.data[:,idx*self.shift+self.period:idx*self.shift+self.period+self.predict_period], self.dates[idx*self.shift:idx*self.shift+self.period],self.dates[idx*self.shift+self.period:idx*self.shift+self.period+self.predict_period]
    def __load_npz__(self):
        l_dir = self.list_dir[::-self.stride]
        l_dir.reverse()
        # l_dir = self._choose_datetime_period(l_dir)
        if self.end is not None:
            l_dir = l_dir[:self.end]
        for i, file in enumerate(l_dir[::self.stride]):
            if i == 0:
                #self.data shape must be like (batch x channels x frames x height x width). here we creeate without batch
                x = np.float32(np.load(self.path_to_dir+f'/{file}', allow_pickle=True))
                if self.resize is not None:
                    x = resize(x, (self.resize[0], self.resize[1]), anti_aliasing=False)
                    self.true_size = x.shape
                else:
                    self.true_size = x.shape
                self.data = np.expand_dims((np.expand_dims(x, axis=0)),axis=0)
                self.dates.append(file.split('_')[1].split('.')[0])
                # if self.pad:
                #     self.pad_shape= (1,1,np.max(self.data.shape),np.max(self.data.shape))
                #     #self.max_size_index = np.array(self.data.shape).argmax()
                #     self.data = np.pad(self.data, [(0,i) for i in np.subtract(self.pad_shape, self.data.shape)], 'constant', constant_values=0)
            else:
                data = np.float32(np.load(self.path_to_dir+f'/{file}', allow_pickle=True))
                if self.resize is not None:
                    data = resize(data, (self.resize[0], self.resize[1]), anti_aliasing=False)
                data = np.expand_dims((np.expand_dims(data, axis=0)),axis=0)
                # if self.pad:
                #     data = np.pad(data, [(0,i) for i in np.subtract(self.pad_shape, data.shape)], 'constant', constant_values=0)
                
                
                self.dates.append(file.split('_')[1].split('.')[0])
                self.data = np.concatenate((self.data, data), axis=1) #Concattanete by frame dimision


def choose_datetime_period(dir_path, from_ymd: Union[List[int], None], to_ymd: Union[List[int], None]) -> List[str]:
    """Function that choose files from dir by date, that define in from_ymd and 
    to_ymd variables.

    Args:
        list_dir list of files names: _description_

    Returns:
        list: _description_
    """
    list_dir = os.listdir(dir_path)
    if from_ymd and to_ymd is None:
        print('No one of date period is chosen! Return all dates!')
        return list_dir
    name = list_dir[0].split('_')[0]
    if from_ymd is not None:  # Choose dates that equal or later than from_ymd.
        list_dir = [file for file in list_dir if datetime(*[datetime.strptime(file, f'{name}_%Y%m%d.npy').year,
                    datetime.strptime(file, f'{name}_%Y%m%d.npy').month,
                    datetime.strptime(file, f'{name}_%Y%m%d.npy').day]) >= datetime(*from_ymd)]
    if to_ymd is not None:  # Choose dates that equal or erlyer than to_ymd.
        list_dir = [file for file in list_dir if datetime(*[datetime.strptime(file, f'{name}_%Y%m%d.npy').year,
                    datetime.strptime(file, f'{name}_%Y%m%d.npy').month,
                    datetime.strptime(file, f'{name}_%Y%m%d.npy').day]) <= datetime(*to_ymd)]
    return sorted(list_dir)

def clip_data_by_batch_and_period(data_list:List[str],
                                  batch_size:int,
                                  period:int,
                                  predict_period:int):
    length_data = len(data_list)
    clip_len = length_data//batch_size*batch_size//(period+predict_period)*(period+predict_period)
    return data_list[:clip_len]


def create_dataloaders(path_to_dir,
                       batch_size: int = 16,
                       shuffle_dataset: bool = False,
                       random_seed: int = 42,
                       in_period: int = 0,
                       predict_period: int = 0,
                       stride: int = 1,
                       test_end: Union[int, None] = None,
                       from_ymd: Union[list, None] = None,
                       to_ymd: Union[list, None] = None,
                       train_test_split: Union[float, None] = 0.2,
                       pad:bool=False,
                       resize_img: Union[List[int], None] = None,
                       shift: int=1
                       ) -> Tuple[Union[torch.utils.data.DataLoader,
                                                                            List[torch.utils.data.DataLoader]],tuple]:
    """
    This function is return two dataloaders (train and test).
    it work for dataset, that consist of train and test data.(in my case i create only one folder with generated data
    and i whant to split Dataset but not folder with data)
    

    Args:
        shift (int): step in predict case. Need to place a predict period step. Defaults to 1.
        path_to_dir (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 16.
        shuffle_dataset (bool, optional): _description_. Defaults to True.
        random_seed (int, optional): _description_. Defaults to 42.
        in_period (int, optional): _description_. Defaults to 0.
        predict_period (int, optional): _description_. Defaults to 0.
        stride (int, optional): _description_. Defaults to 1.
        test_end (Union[int, None], optional): _description_. Defaults to None.
        from_ymd (Union[list, None], optional): _description_. Defaults to None.
        to_ymd (Union[list, None], optional): _description_. Defaults to None.
        train_test_split (Union[float, None], optional): _description_. Defaults to 0.2.
        pad (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[Union[torch.utils.data.DataLoader, List[torch.utils.data.DataLoader]],tuple]: return ether Train/tets dataloader or full dataloader. 
        And Return tuple of image true size
    """
    list_dir = choose_datetime_period(dir_path=path_to_dir,
                                      from_ymd=from_ymd,
                                      to_ymd=to_ymd)
    if train_test_split is not None:
        length = len(list_dir)
        train_part = int((1-train_test_split)*length)
        train_data, test_data = list_dir[:train_part], list_dir[train_part:]
        if len(list_dir[train_part::stride])//(predict_period+in_period)==0 or len(list_dir[:train_part:stride])//(predict_period+in_period)==0:
            print('You choose too short date period! You may choose bigger period between from_ymd and to_ymd. Or choose lower periods for predicts')
            print('Now chosen date from:',from_ymd,'To:',to_ymd)
        while len(list_dir[train_part::stride])//(predict_period+in_period)==0 or len(list_dir[:train_part:stride])//(predict_period+in_period)==0:
            new_f = datetime(*from_ymd)-timedelta(days=stride*(predict_period+in_period)//2)
            new_to = datetime(*to_ymd)+timedelta(days=stride*(predict_period+in_period)//2)
            from_ymd = [new_f.year,new_f.month,new_f.day]
            to_ymd = [new_to.year,new_to.month,new_to.day]

            list_dir = choose_datetime_period(dir_path=path_to_dir,
                                      from_ymd=from_ymd,
                                      to_ymd=to_ymd)
            length = len(list_dir)
            train_part = int((1-train_test_split)*length)
            train_data, test_data = list_dir[:train_part], list_dir[train_part:]
        print("After validation date period is: From:",from_ymd,'To:',to_ymd)
        train_data = clip_data_by_batch_and_period(data_list=train_data,
                                                   batch_size=batch_size,
                                                   period=in_period,
                                                   predict_period=predict_period)
        test_data = clip_data_by_batch_and_period(data_list=test_data,
                                                  batch_size=batch_size,
                                                  period=in_period,
                                                  predict_period=predict_period)
        train_dataset = IceDataset(list_dir=train_data,
                                   path_to_dir=path_to_dir,
                                   in_period=in_period,
                                   predict_period=predict_period,
                                   stride=stride,
                                   test_end=test_end,
                                   from_ymd=from_ymd,
                                   to_ymd=to_ymd,
                                   pad=pad,
                                   resize_img=resize_img,
                                   shift=shift)
        test_dataset = IceDataset(list_dir=test_data,
                                  path_to_dir=path_to_dir,
                                  in_period=in_period,
                                  predict_period=predict_period,
                                  stride=stride,
                                  test_end=test_end,
                                  from_ymd=from_ymd,
                                  to_ymd=to_ymd,
                                  pad=pad,
                                  resize_img=resize_img,
                                  shift=shift)
        train_true_size = train_dataset.true_size
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                      drop_last=False,
                                                      shuffle=shuffle_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=batch_size,
                                                     drop_last=False)
        return [train_loader, test_loader],train_true_size
    else:
        dataset = IceDataset(list_dir=list_dir,
                             path_to_dir=path_to_dir,
                             in_period=in_period,
                             predict_period=predict_period,
                             stride=stride,
                             test_end=test_end,
                             from_ymd=from_ymd,
                             to_ymd=to_ymd,
                             pad=pad,
                             resize_img=resize_img,
                             shift=shift)
        true_size = dataset.true_size
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=False)
        return loader,true_size

if __name__=='__main__':
    dataloaders,sizes = create_dataloaders(path_to_dir=r'laptev',
                                    batch_size=1,
                                    in_period=2,
                                    predict_period=2,
                                    stride=7,
                                    test_end=None,
                                    from_ymd=[2012, 1, 1],
                                    to_ymd=[2013, 4, 4],
                                    pad=False,
                                    resize_img=[55,65],
                                    train_test_split=None)
####
    for train in dataloaders:

        print(train[0].shape,train[1].shape)

    for test in dataloaders[1]:
        print(test[0].shape,test[1].shape)

