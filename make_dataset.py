import numpy as np
import os
import pickle
from tqdm import tqdm

dir_train = 'inputs/train/'
dir_test = 'inputs/test/'

def original_set(dir_train, dir_test):

    # train dataset
    train = []
    train_y = []

    for i in os.listdir(dir_train):
        npy = np.load(dir_train + i)
        
        # missing value 제거
        if npy[:,:,-1].sum() < 0:
            continue
        if (npy[:,:,-1] >= 0.1).sum() >= 50:
            train.append(npy[:,:,:-1])
            train_y.append(npy[:,:,-1])

    train = np.array(train)
    train_y = np.array(train_y)

    with open('inputs/train50.pickle', 'wb') as f:
        pickle.dump(train, f, protocol=4)

    with open('inputs/train_y50.pickle', 'wb') as f:
        pickle.dump(train_y, f, protocol=4)

    del train
    del train_y

    # test dataset
    # test = []

    # for i in os.listdir(dir_test):
    #     npy = np.load(dir_test + i)
    #     test.append(npy)
    # test = np.array(test)

    # with open('inputs/test.pickle', 'wb') as f:
    #     pickle.dump(test, f, protocol=4)
    # del test



def augmentation(dir_train):

# Augmentation (rotation 데이터 만들기)
# - 데이터에 강수량이 존재하는 자료만 rotation 자료 만들 것 - 73000여개 중 61849개에서 비가 옴
# - 해당 사진에서 평균적으로 0.1 이상인 것만 rotation 하거나 (대회 평가에서 0.1 이상인 픽셀만 사용한다고 했음)
# - 1600픽셀 중 강수량이 0.1 이상인 픽셀이 100개 이상인 사진만 로테이션 시켜도 되는데 우선은 후자로 테스트 해보자.

    rot_train = []
    rot_train_y = []

    for i in tqdm(os.listdir(dir_train)):
        npy = np.load(dir_train + i)
        
        if npy[:,:,-1].sum() < 0:
            continue
        
        # 1600 픽셀 사진에서 강수량이 0.1 이상인 픽셀이 100개 이상인 사진만 로테이션 시켜보자.
        if (npy[:,:,-1] >= 0.1).sum() > 100:
            npy = np.rot90(npy, 3, (0,1)) # 2번째인자가 1이면 90도, 2이면 180도, 3번째 인자는 어떤 축을 잡고 돌릴것인지, 0번축, 1번축을 기준으로 돌릴것이다
            rot_train.append(npy[:,:,:-1])
            rot_train_y.append(npy[:,:,-1])
        
    rot_train = np.array(rot_train)
    rot_train_y = np.array(rot_train_y)

    with open('inputs/rot_train270.pickle', 'wb') as f:
        pickle.dump(rot_train, f, protocol=4)

    with open('inputs/rot_train270_y.pickle', 'wb') as f:
        pickle.dump(rot_train_y, f, protocol=4)
    
    del rot_train
    del rot_train_y
    

def augmentation_to_npy(dir_train):

# Augmentation (rotation 데이터 만들기)
# - 데이터에 강수량이 존재하는 자료만 rotation 자료 만들 것 - 73000여개 중 61849개에서 비가 옴
# - 해당 사진에서 평균적으로 0.1 이상인 것만 rotation 하거나 (대회 평가에서 0.1 이상인 픽셀만 사용한다고 했음)
# - 1600픽셀 중 강수량이 0.1 이상인 픽셀이 100개 이상인 사진만 로테이션 시켜도 되는데 우선은 후자로 테스트 해보자.

    rot = {
        1: '90',
        2: '180',
        3: '270'
    }

    for i in tqdm(os.listdir(dir_train)):
        npy = np.load(dir_train + i)
        
        if npy[:,:,-1].sum() < 0:
            continue
        
        # 1600 픽셀 사진에서 강수량이 0.1 이상인 픽셀이 100개 이상인 사진만 로테이션 시켜보자.
        if (npy[:,:,-1] >= 0.1).sum() > 100:
            nplr = np.fliplr(npy)
            npud = np.flipud(npy)
            np.save(f'inputs/rot/{i}_lr.npy', nplr)
            np.save(f'inputs/rot/{i}_ud.npy', npud)
            for angle in range(1, 4):
                npy_ = np.rot90(npy, angle, (0,1)) # 2번째인자가 1이면 90도, 2이면 180도, 3번째 인자는 어떤 축을 잡고 돌릴것인지, 0번축, 1번축을 기준으로 돌릴것이다
                np.save(f'inputs/rot/{i}_{rot[angle]}.npy', npy_)
            

original_set(dir_train, dir_test)
# augmentation(dir_train)
# augmentation_to_npy(dir_train)