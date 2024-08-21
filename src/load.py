import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE
from typing import List


def get_column_indices_by_type(df: pd.DataFrame, type: str) -> List[int]:
    col_types = df.dtypes.reset_index(drop=True)
    col_indices = col_types[col_types == type].index
    return list(col_indices)


def _get_upsampler(
    features: pd.DataFrame,
    random_state: int
) -> SMOTE | SMOTENC:
    category_cols = features.select_dtypes(['O', 'category']).columns
    if len(category_cols) > 0:
        print(" => SMOTENC method implemented")
        col_indices = get_column_indices_by_type(features, "category")
        return SMOTENC(
            random_state=random_state,
            categorical_features=col_indices
        )

    else:
        print(" => SMOTE method implemented")
        return SMOTE(random_state=random_state)


def _upsample_data(
    df: pd.DataFrame,
    target: str,
    random_state: int
) -> pd.DataFrame:
    X = df.drop([target], axis=1)
    y = df[target]
    upsampler = _get_upsampler(features=X, random_state=random_state)
    X_res, y_res = upsampler.fit_resample(X, y)
    return pd.concat([X_res, y_res], axis=1)


def upsample_data(
    df: pd.DataFrame,
    target: str,
    random_state: int
) -> pd.DataFrame:
    print("\nBeginning Upsampling:")
    print(" => Intial distribution:")
    print(df[target].value_counts())
    upsample = _upsample_data(df=df, target=target, random_state=random_state)
    print(" => Upsampled distribution:")
    print(upsample[target].value_counts())
    print(" => Upsampling complete")
    return upsample


def check_column_bounds(
    c_min: int | float,
    c_max: int | float,
    type: np.int8 | np.int16 | np.int32 | np.int64 | np.float16 | np.float32
) -> bool:
    if type.__name__[:3] == 'int':
        return c_min > np.iinfo(type).min and c_max < np.iinfo(type).max
    else:
        return c_min > np.finfo(type).min and c_max < np.finfo(type).max


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    if len(df.select_dtypes('category').columns) > 0:
        print("Data is already reduced\n")
        return df

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if check_column_bounds(c_min, c_max, np.int8):
                    df[col] = df[col].astype(np.int8)
                elif check_column_bounds(c_min, c_max, np.int16):
                    df[col] = df[col].astype(np.int16)
                elif check_column_bounds(c_min, c_max, np.int32):
                    df[col] = df[col].astype(np.int32)
                elif check_column_bounds(c_min, c_max, np.int64):
                    df[col] = df[col].astype(np.int64)
            else:
                if check_column_bounds(c_min, c_max, np.float16):
                    df[col] = df[col].astype(np.float16)
                elif check_column_bounds(c_min, c_max, np.float32):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    reduction_value = 100 * (start_mem - end_mem) / start_mem
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%\n'.format(reduction_value))

    return df
