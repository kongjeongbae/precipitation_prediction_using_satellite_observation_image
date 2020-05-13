import numpy as np
import gc

def rotation(train, test):
    all_np = np.vstack([train, test])
    LEN_TRAIN = len(train)
    del train, test

    rotation_45_1_1 = all_np[:, :, :, -4] + all_np[:, :, :, -3]
    rotation_45_1_2 = all_np[:, :, :, -4] - all_np[:, :, :, -3]
    rotation_45_2_1 = all_np[:, :, :, -2] + all_np[:, :, :, -1]
    rotation_45_2_2 = all_np[:, :, :, -2] - all_np[:, :, :, -1]

    rotation_45_1_1 = rotation_45_1_1.reshape(all_np.shape[0], 40, 40,1)
    rotation_45_1_2 = rotation_45_1_2.reshape(all_np.shape[0], 40, 40,1)
    rotation_45_2_1 = rotation_45_2_1.reshape(all_np.shape[0], 40, 40,1)
    rotation_45_2_2 = rotation_45_2_2.reshape(all_np.shape[0], 40, 40,1)

    all_np = np.concatenate((all_np, rotation_45_1_1), axis=-1)
    all_np = np.concatenate((all_np, rotation_45_1_2), axis=-1)
    all_np = np.concatenate((all_np, rotation_45_2_1), axis=-1)
    all_np = np.concatenate((all_np, rotation_45_2_2), axis=-1)

    del rotation_45_1_1
    del rotation_45_1_2
    del rotation_45_2_1
    del rotation_45_2_2


    rotation_30_1_1 = all_np[:, :, :, -4] * np.cos(np.pi / 6) + all_np[:, :, :, -3] * np.sin(np.pi / 6)
    rotation_30_1_2 = all_np[:, :, :, -4] * np.cos(np.pi / 6) - all_np[:, :, :, -3] * np.sin(np.pi / 6)
    rotation_30_2_1 = all_np[:, :, :, -2] * np.cos(np.pi / 6) + all_np[:, :, :, -1] * np.sin(np.pi / 6)
    rotation_30_2_2 = all_np[:, :, :, -2] * np.cos(np.pi / 6) - all_np[:, :, :, -1] * np.sin(np.pi / 6)

    rotation_30_1_1 = rotation_30_1_1.reshape(all_np.shape[0], 40, 40,1)
    rotation_30_1_2 = rotation_30_1_2.reshape(all_np.shape[0], 40, 40,1)
    rotation_30_2_1 = rotation_30_2_1.reshape(all_np.shape[0], 40, 40,1)
    rotation_30_2_2 = rotation_30_2_2.reshape(all_np.shape[0], 40, 40,1)

    all_np = np.concatenate((all_np, rotation_30_1_1), axis=-1)
    all_np = np.concatenate((all_np, rotation_30_1_2), axis=-1)
    all_np = np.concatenate((all_np, rotation_30_2_1), axis=-1)
    all_np = np.concatenate((all_np, rotation_30_2_2), axis=-1)

    del rotation_30_1_1
    del rotation_30_1_2
    del rotation_30_2_1
    del rotation_30_2_2


    rotation_60_1_1 = all_np[:, :, :, -4] * np.cos(np.pi / 3) + all_np[:, :, :, -3] * np.sin(np.pi / 3)
    rotation_60_1_2 = all_np[:, :, :, -4] * np.cos(np.pi / 3) - all_np[:, :, :, -3] * np.sin(np.pi / 3)
    rotation_60_2_1 = all_np[:, :, :, -2] * np.cos(np.pi / 3) + all_np[:, :, :, -1] * np.sin(np.pi / 3)
    rotation_60_2_2 = all_np[:, :, :, -2] * np.cos(np.pi / 3) - all_np[:, :, :, -1] * np.sin(np.pi / 3)

    rotation_60_1_1 = rotation_60_1_1.reshape(all_np.shape[0], 40, 40,1)
    rotation_60_1_2 = rotation_60_1_2.reshape(all_np.shape[0], 40, 40,1)
    rotation_60_2_1 = rotation_60_2_1.reshape(all_np.shape[0], 40, 40,1)
    rotation_60_2_2 = rotation_60_2_2.reshape(all_np.shape[0], 40, 40,1)

    all_np = np.concatenate((all_np, rotation_60_1_1), axis=-1)
    all_np = np.concatenate((all_np, rotation_60_1_2), axis=-1)
    all_np = np.concatenate((all_np, rotation_60_2_1), axis=-1)
    all_np = np.concatenate((all_np, rotation_60_2_2), axis=-1)

    del rotation_60_1_1
    del rotation_60_1_2
    del rotation_60_2_1
    del rotation_60_2_2


    train = all_np[:LEN_TRAIN]
    test = all_np[LEN_TRAIN:]
    del all_np
    gc.collect()

    return train, test


def terrain(train, test):
    a = (train[:, :, :, 9] / 100)
    a = a.astype('int8')
    a = a.reshape(75957, 40, 40, 1)
    train = np.concatenate((train, a), axis=-1)

    a = (test[:, :, :, 9] / 100)
    a = a.astype('int8')
    a = a.reshape(75957, 40, 40, 1)
    test = np.concatenate((test, a), axis=-1)
    del a

    return train, test