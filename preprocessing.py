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


def rotation(data):
    v1_m_h1 = data[:, :, :, 0] * np.cos(np.pi / 4) - data[:, :, :, 1] * np.sin(np.pi / 4)  
    v1_p_h1 = data[:, :, :, 0] * np.cos(np.pi / 4) + data[:, :, :, 1] * np.sin(np.pi / 4)  
    data[:, :, :, 0] = v1_m_h1
    data[:, :, :, 1] = v1_p_h1 
    del v1_m_h1
    del v1_p_h1

    v2_m_h2 = data[:, :, :, 2] * np.cos(np.pi / 4) - data[:, :, :, 3] * np.sin(np.pi / 4)
    v2_p_h2 = data[:, :, :, 2] * np.cos(np.pi / 4) + data[:, :, :, 3] * np.sin(np.pi / 4)
    data[:, :, :, 2] = v2_m_h2
    data[:, :, :, 3] = v2_p_h2 
    del v2_m_h2
    del v2_p_h2
    
    v4_p_h4_30 = data[:, :, :, 5] * np.cos(np.pi / 4) + data[:, :, :, 6] * np.sin(np.pi / 4)
    v4_m_h4_30 = data[:, :, :, 5] * np.cos(np.pi / 4) - data[:, :, :, 6] * np.sin(np.pi / 4)
    data[:, :, :, 5] = v4_p_h4_30
    data[:, :, :, 6] = v4_m_h4_30

    v5_p_h5_30 = data[:, :, :, 7] * np.cos(np.pi / 6) + data[:, :, :, 8] * np.sin(np.pi / 6)
    v5_m_h5_30 = data[:, :, :, 7] * np.cos(np.pi / 6) - data[:, :, :, 8] * np.sin(np.pi / 6)
    data[:, :, :, 7] = v5_p_h5_30
    data[:, :, :, 8] = v5_m_h5_30

    del v4_p_h4_30
    del v4_m_h4_30
    del v5_p_h5_30
    del v5_m_h5_30
    
    return data