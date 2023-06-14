# stdlib
from typing import Any, List, Optional, Tuple

# third party
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# adjutorium absolute
import adjutorium.logger as log
from adjutorium.plugins.imputers import Imputers
from adjutorium.plugins.preprocessors import Preprocessors

CATEGORICAL_THRESHOLD = 10
ONEHOT_ENCODE_THRESHOLD = 3


def _fillna(df: pd.DataFrame, column: str) -> pd.Series:
    dummy: Any = "NaN"
    if is_numeric_dtype(df[column]):
        dummy = -9999

    return df[column].fillna(dummy)


class EncodersCallbacks:
    def __init__(self, encoders: dict, imputer: Any = None) -> None:
        self.encoders = encoders
        self.imputer = imputer

    def set_imputer(self, imputer: Any) -> None:
        self.imputer = imputer

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for col in self.encoders:
            enc = self.encoders[col]
            target = _fillna(output, col)
            if hasattr(enc, "get_feature_names"):
                # onehot encoder
                encoded = pd.DataFrame(
                    enc.transform(target.values.reshape(-1, 1)),
                    columns=enc.get_feature_names([col]),
                    index=output.index.copy(),
                )

            else:

                # label encoder
                encoded = pd.DataFrame(
                    enc.transform(target),
                    columns=[col],
                    index=output.index.copy(),
                )

            output.drop(columns=[col], inplace=True)
            output = pd.concat([output, encoded], axis=1)

        if self.imputer:
            columns = output.columns
            output = self.imputer.transform(output)
            output.columns = columns

            assert not output.isnull().values.any()

        return output

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for col in self.encoders:
            enc = self.encoders[col]
            if hasattr(enc, "get_feature_names"):
                columns = enc.get_feature_names([col])
            else:
                columns = [col]

            decoded = pd.DataFrame(
                enc.inverse_transform(output[columns].astype(int).values),
                columns=[col],
                index=output.index.copy(),
            )

            output.drop(columns=columns, inplace=True)
            output = pd.concat([output, decoded], axis=1)

        assert not output.isnull().values.any()
        return output

    def numeric_decode(self, df: pd.DataFrame, strategy: str = "max") -> pd.DataFrame:
        output = df.copy()
        for col in self.encoders:
            enc = self.encoders[col]
            if hasattr(enc, "get_feature_names"):
                columns = enc.get_feature_names([col])
            else:
                columns = [col]
            if strategy == "max":
                vals = output[columns].max(axis=1)
            else:
                raise ValueError(f"unknown strategy {strategy}")
            output.drop(columns=columns, inplace=True)
            output[col] = vals
        assert not output.isnull().values.any()
        return output

    def __getitem__(self, key: str) -> Any:
        return self.encoders[key]


def dataframe_hash(df: pd.DataFrame) -> str:
    return str(abs(pd.util.hash_pandas_object(df).sum()))


def dataframe_remove_zeros(df: pd.DataFrame, column: str) -> pd.DataFrame:
    keep = df[column] > 0
    return df[keep]


def dataframe_imputation(
    df: pd.DataFrame, method: str = "ice"
) -> Tuple[pd.DataFrame, Any]:
    log.info(f"preprocess: dataset imputation using {method}")
    columns = df.columns
    imputer = Imputers().get(method)

    output = imputer.fit_transform(df)
    output.columns = columns

    output.reset_index(drop=True, inplace=True)
    return output, imputer


def dataframe_encode(in_df: pd.DataFrame) -> Tuple[pd.DataFrame, EncodersCallbacks]:
    df = in_df.copy()
    encoders = {}

    def _is_categorical(column: str) -> bool:
        target = df[column]
        dtype = target.dtype
        unique_cnt = len(target.unique())

        return dtype == "object" or unique_cnt < CATEGORICAL_THRESHOLD

    def _is_missing(column: str) -> bool:
        return df[column].isnull().sum() != 0

    # MISSING CATEGORICAL -> ENCODE WITH SPECIAL VALUE
    # CATEGORICAL: ENCODE
    # MISSING: IMPUTE

    for column in df.columns:
        if not _is_categorical(column):
            log.info(f"handling continuous column {column}")
            continue

        if _is_missing(column):
            log.info(f"handling categorical column with missingness {column}")
            target = _fillna(df, column)
        else:
            log.info(f"handling categorical column {column}")
            target = df[column]

        unique_cnt = len(target.unique())

        if unique_cnt > ONEHOT_ENCODE_THRESHOLD:
            log.info(f"preprocess: dataset label encoding for {column}")
            le = LabelEncoder()

            target = pd.Series(le.fit_transform(target), index=df[column].index.copy())
            df.loc[:, column] = target

            encoders[column] = le
        else:
            log.info(f"preprocess: dataset one-hot encoding for {column}")
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            target = target.values.reshape(-1, 1)
            ohe.fit(target)
            encoders[column] = ohe

            encoded = pd.DataFrame(
                ohe.transform(target),
                columns=ohe.get_feature_names([column]),
                index=df.index.copy(),
            )
            df = pd.concat([df, encoded], axis=1)
            df.drop(columns=[column], inplace=True)

    return df, EncodersCallbacks(encoders)


def dataframe_encode_and_impute(
    orig_df: pd.DataFrame, imputation_method: Optional[str]
) -> Tuple[pd.DataFrame, EncodersCallbacks]:
    df = orig_df.copy()
    df, encoder_ctx = dataframe_encode(df)

    if df.isnull().values.any() and imputation_method:
        df, imputer = dataframe_imputation(df, imputation_method)
        encoder_ctx.set_imputer(imputer)

    return df, encoder_ctx


def dataframe_sample(X: pd.DataFrame, Y: pd.DataFrame, max_size: int = 5000) -> List:
    log.info(f"preprocess: dataset subsampling {max_size}")
    df_limit = len(Y.unique()) * max_size
    ratio = df_limit / len(X)

    if ratio >= 1:
        return list(range(len(X)))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=0)
    for _, sample_index in sss.split(X, Y):
        return sample_index

    raise ValueError("should not be here")


def dataframe_drop_low_variance(df: pd.DataFrame) -> pd.DataFrame:
    plugin = Preprocessors(category="dimensionality_reduction").get(
        "variance_threshold"
    )

    output = plugin.fit_transform(df)
    cols_dropped = [col for col in df.columns if col not in output.columns]
    if len(cols_dropped) != 0:
        log.info(f"preprocess: {cols_dropped} were removed due to low variance")

    return output


def dataframe_preprocess(
    df: pd.DataFrame,
    target: str,
    time_to_event: Optional[str] = None,
    special_cols: List[str] = [],
    sample: bool = True,
    imputation_method: Optional[str] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    List,
    EncodersCallbacks,
]:

    drop_columns = [target]

    others = []

    T = None
    if time_to_event is not None:
        df = dataframe_remove_zeros(df, time_to_event)
        df = df[df[time_to_event] > 0]
        drop_columns.append(time_to_event)

        T = df[time_to_event]

    for col in special_cols:
        df = dataframe_remove_zeros(df, col)
        drop_columns.append(col)

        others.append(df[col])

    X = df.drop(drop_columns, axis=1)
    Y = df[target]

    (
        X,
        encoders,
    ) = dataframe_encode_and_impute(X, imputation_method)

    X = dataframe_drop_low_variance(X)

    if sample:
        indices = dataframe_sample(X, Y)

        X = X.loc[X.index[indices]]
        Y = Y.loc[Y.index[indices]]
        if T is not None:
            T = T.loc[T.index[indices]]

        for idx, other in enumerate(others):
            others[idx] = other.loc[other.index[indices]]

    return X, T, Y, others, encoders
