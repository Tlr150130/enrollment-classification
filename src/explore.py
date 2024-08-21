import math
import pandas as pd
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt


def calculate_plot_columns(df: pd.DataFrame, column: str) -> int:
    n_unique = len(df[column].unique())
    if n_unique <= 3:
        return n_unique
    return math.ceil(math.sqrt(n_unique))


def _explore_plot_category(
    df: pd.DataFrame,
    col: str,
    target: str,
    height: float | None,
    aspect: float | None
) -> None:
    if height is None:
        height = 2.5
    if aspect is None:
        aspect = 1.0

    with sns.axes_style(style='ticks'):
        n = calculate_plot_columns(df, col)
        sns.catplot(
            x=target,
            col=col,
            hue=target,
            col_wrap=n,
            data=df,
            kind="count",
            height=height,
            aspect=aspect
        )
    return None


def _explore_plot_int(
    df: pd.DataFrame,
    col: str,
    target: str,
    min_val: float | None,
    max_val: float | None,
    aspect: float | None
) -> None:
    with sns.axes_style('white'):
        if min_val is None:
            min_val = math.floor(df[col].min())
        if max_val is None:
            max_val = math.ceil(df[col].max()) + 1
        if aspect is None:
            aspect = 3.0

        g = sns.catplot(
            x=col,
            data=df,
            aspect=aspect,
            kind='count',
            hue=target,
            order=range(min_val, max_val))
        g.set_ylabels(f'{col} vs {target}')
    return None


def _explore_plot_float(
    df: pd.DataFrame,
    col: str,
    target: str,
    min_val: float | None,
    max_val: float | None
) -> None:
    plotting_data = df.copy()
    if min_val is not None:
        plotting_data = plotting_data[plotting_data[col] >= min_val]
    if max_val is not None:
        plotting_data = plotting_data[plotting_data[col] <= max_val]

    sns.displot(data=plotting_data, x=col, hue=target, kind="kde")
    plt.title(f"{col} vs. {target}")
    return None


def _explore_plot_numeric(
    df: pd.DataFrame,
    col: str,
    target: str,
    min_val: float | None,
    max_val: float | None,
    aspect: float | None
) -> None:
    type = str(df[col].dtype)
    if 'int' in type:
        _explore_plot_int(
            df=df,
            col=col,
            target=target,
            min_val=min_val,
            max_val=max_val,
            aspect=aspect
        )
    elif 'float' in type:
        _explore_plot_float(
            df=df,
            col=col,
            target=target,
            min_val=min_val,
            max_val=max_val
        )
    else:
        raise TypeError(f"Unexpected type encountered: {type}")
    return None


def explore_single_plot(
    df: pd.DataFrame,
    col: str,
    target: str,
    min_val: float | None,
    max_val: float | None,
    height: float | None,
    aspect: float | None
) -> None:
    if df[col].dtype == 'category':
        _explore_plot_category(
            df=df,
            col=col,
            target=target,
            height=height,
            aspect=aspect
        )
    else:
        _explore_plot_numeric(
            df=df,
            col=col,
            target=target,
            min_val=min_val,
            max_val=max_val,
            aspect=aspect
        )
    return None


def explore_cat_cat_plot(
    df: pd.DataFrame,
    x: str,
    grouping: str,
    target: str,
    height: float | None,
    aspect: float | None
) -> None:
    with sns.axes_style('white'):
        n = calculate_plot_columns(df, grouping)
        sns.catplot(
            x=x,
            hue=target,
            col=grouping,
            data=df,
            kind="count",
            col_wrap=n,
            height=height,
            aspect=aspect
        )


def explore_cat_num_plot(
    df: pd.DataFrame,
    y: str,
    grouping: str,
    target: str,
    min_val: float | None,
    max_val: float | None,
    height: float | None,
    aspect: float | None
) -> None:
    plotting_data = df.copy()
    if min_val is not None:
        plotting_data = plotting_data[plotting_data[y] >= min_val]
    if max_val is not None:
        plotting_data = plotting_data[plotting_data[y] <= max_val]

    with sns.axes_style('white'):
        n = calculate_plot_columns(plotting_data, grouping)
        sns.catplot(
            x=target,
            y=y,
            col=grouping,
            hue=target,
            data=plotting_data,
            kind="box",
            col_wrap=n,
            height=height,
            aspect=aspect
        )


def explore_num_num_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    target: str,
    min_val: List[float | None] | None,
    max_val: List[float | None] | None
) -> None:
    plotting_data = df.copy()
    if min_val is not None:
        if not isinstance(min_val, list):
            raise TypeError("Min values needs to be a list")
        if len(min_val) != 2:
            raise IndexError("Min values requires 2 values")
        if min_val[0] is not None:
            plotting_data = plotting_data[plotting_data[x] >= min_val[0]]
        if min_val[1] is not None:
            plotting_data = plotting_data[plotting_data[y] >= min_val[1]]
    if max_val is not None:
        if not isinstance(max_val, list):
            raise TypeError("Max values needs to be a list")
        if len(max_val) != 2:
            raise IndexError("Max values requires 2 values")
        if max_val[0] is not None:
            plotting_data = plotting_data[plotting_data[x] <= max_val[0]]
        if max_val[1] is not None:
            plotting_data = plotting_data[plotting_data[y] <= max_val[1]]

    sns.scatterplot(
        data=plotting_data,
        x=x,
        y=y,
        hue=target,
        size=target
    )


def explore_multi_plot(
    df: pd.DataFrame,
    cols: str,
    target: str,
    min_val: float | None,
    max_val: float | None,
    height: float | None,
    aspect: float | None
) -> None:
    col1_type = df[cols[0]].dtype.name[:3]
    col2_type = df[cols[1]].dtype.name[:3]

    if col1_type == 'cat' and col2_type == 'cat':
        explore_cat_cat_plot(
            df=df,
            x=cols[0],
            grouping=cols[1],
            target=target,
            height=height,
            aspect=aspect
        )

    elif col1_type == 'cat' and col2_type != 'cat':
        explore_cat_num_plot(
            df=df,
            y=cols[1],
            grouping=cols[0],
            target=target,
            min_val=min_val,
            max_val=max_val,
            height=height,
            aspect=aspect
        )

    elif col1_type != 'cat' and col2_type == 'cat':
        explore_cat_num_plot(
            df=df,
            y=cols[0],
            grouping=cols[1],
            target=target,
            min_val=min_val,
            max_val=max_val,
            height=height,
            aspect=aspect
        )

    else:
        explore_num_num_plot(
            df=df,
            x=cols[0],
            y=cols[1],
            target=target,
            min_val=min_val,
            max_val=max_val
        )

    return None


def explore_plot(
    df: pd.DataFrame,
    cols: str | List[str],
    target: str,
    min_val: float | None = None,
    max_val: float | None = None,
    height: float | None = None,
    aspect: float | None = None
) -> None:
    if isinstance(cols, list):
        n_cols = len(cols)
        if n_cols == 0:
            raise IndexError("Cols must contain column names")
        if n_cols == 1:
            explore_single_plot(
                df=df,
                col=cols[0],
                target=target,
                min_val=min_val,
                max_val=max_val,
                height=height,
                aspect=aspect
            )
        elif n_cols == 2:
            explore_multi_plot(
                df=df,
                cols=cols,
                target=target,
                min_val=min_val,
                max_val=max_val,
                height=height,
                aspect=aspect
            )
        else:
            raise IndexError("Max 2 columns can be passed")
    else:
        explore_single_plot(
            df=df,
            col=cols,
            target=target,
            min_val=min_val,
            max_val=max_val,
            height=height,
            aspect=aspect
        )
    return None


def describe_categorical_features(data: pd.DataFrame) -> None:
    n = len(data)
    for index, col in enumerate(data.select_dtypes('category').columns):
        print(f"{index}. {col}")
        values = []
        for value in data[col].unique():
            count = sum(data[col] == value)
            percentage = round(count/n*100, 2)
            values.append([value, percentage])

        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
        sorted_strings = [
            f"{value[0]} ({value[1]}%)" for value in sorted_values]
        print(f" => {', '.join(sorted_strings)}")
