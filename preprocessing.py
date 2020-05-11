from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import numpy as np
import gc

def scaling(train, test, how):
    TR_SHAPE_0, TR_SHAPE_1, TR_SHAPE_2, TR_SHAPE_3 = train.shape
    TE_SHAPE_0, TE_SHAPE_1, TE_SHAPE_2, TE_SHAPE_3 = test.shape
    train = train.reshape(TR_SHAPE_0 * TR_SHAPE_1 * TR_SHAPE_2, TR_SHAPE_3)
    test = test.reshape(TE_SHAPE_0 * TE_SHAPE_1 * TE_SHAPE_2, TE_SHAPE_3)
    all_np = np.vstack([train, test])

    del train
    del test
    gc.collect()
    # all_np = ss.fit_transform(all_np)

    if how == 'rs':
        print('RobustScaler')
        for i in range(all_np.shape[1]):
            rs = RobustScaler()
            all_np[:,i]  = rs.fit_transform(all_np[:,i].reshape(-1, 1)).reshape(len(all_np))

    elif how == 'ms':
        print('MinMaxScaler')
        for i in range(all_np.shape[1]):
            ms = MinMaxScaler()
            all_np[:,i]  = ms.fit_transform(all_np[:,i].reshape(-1, 1)).reshape(len(all_np))

    else:
        for i in range(all_np.shape[1]):
            ss = StandardScaler()
            all_np[:,i]  = ss.fit_transform(all_np[:,i].reshape(-1, 1)).reshape(len(all_np))


    train = all_np[:TR_SHAPE_0 * TR_SHAPE_1 * TR_SHAPE_2]
    test = all_np[TR_SHAPE_0 * TR_SHAPE_1 * TR_SHAPE_2:]

    del all_np

    train = train.reshape(TR_SHAPE_0 , TR_SHAPE_1 , TR_SHAPE_2, TR_SHAPE_3)
    test = test.reshape(TE_SHAPE_0 , TE_SHAPE_1 , TE_SHAPE_2, TE_SHAPE_3)

    return train, test