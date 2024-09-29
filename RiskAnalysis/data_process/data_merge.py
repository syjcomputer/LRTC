import pandas as pd

def datamerge(out_path, base_risk_num, paths, types='coulmn'):
    '''
    :param out_path:
    :param base_risk_num:
    :param paths:
    :param types: coulmn or row
    :return:
    '''
    cols = []
    dfs = []

    if types == 'row':
        for i in range(0, len(paths)):
            df = pd.read_csv(paths[i])
            dfs.append(df)
        all_csv = pd.concat(dfs)

    else:
        for i in range(base_risk_num):
            cols.append(i + 2)
        df = pd.read_csv(paths[0])
        dfs.append(df)

        for i in range(1, len(paths)):
            df = pd.read_csv(paths[i], usecols=cols)
            dfs.append(df)
        all_csv = pd.concat(dfs, axis=1)

    all_csv.to_csv(out_path, index=False)
