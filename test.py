import pandas as pd
from mean_encoding import MeanEncoding
from sklearn.compose import ColumnTransformer


df = pd.DataFrame({'a1': ['q', 'w', 'e', 'r', 'q', 'w', 'e', 'r'],
                   'a2': [3, 3, 3, 3, 4, 4, 4, 4],
                   'c': [3, 3, 4, 3, 2, 2, 4, 4],
                   'd': [3, 3, 4, 3, 2, 2, 4, 4],
                   'b': [1, 0, 1, 0, 1, 0, 1, 1]})

df.set_index(['a1', 'a2'], inplace=True, drop=True)

preprocessor = ColumnTransformer(transformers=[('mean_enc',
                                                MeanEncoding(target='b', strategy='mean_reg', alpha=0, seed=1),
                                                ['c', 'd', 'b'])],
                                 remainder="passthrough")

preprocessor.fit(df)
a = preprocessor.transform(df)
print(a)
