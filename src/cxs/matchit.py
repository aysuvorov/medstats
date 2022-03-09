import pandas as pd
import numpy as np

import rpy2.robjects as ro

from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
env = ro.r.globalenv()

from rpy2.robjects import pandas2ri, FloatVector, IntVector, FactorVector, Formula, r

import rpy2.robjects.numpy2ri as rpyn
rpyn.activate()
stats = importr('stats')
base = importr('base')
psm = importr('MatchIt')

#+---------------------------------------------------

def matchit(df, col_lst,ratio, formula, id):
    tb = df[col_lst].copy()
    for col in [x for x in tb.columns if pd.CategoricalDtype.is_dtype(tb[x]) == True]:
        tb[col] = tb[col].astype(str)
        tb[col] = tb[col].astype('category')

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(tb)

    del tb

    form = Formula(formula)
    f = stats.glm(formula = form, family = 'binomial', data = r_df)
    r_df.scores = f[2]

    match1 = psm.matchit(formula = form, ratio = ratio, data=r_df)
    m_df = psm.match_data(match1, drop_unmatched = True)
    tb = pd.DataFrame(ro.conversion.rpy2py(m_df))

    for col in tb.columns:
        if tb[col].dtypes == 'object':
            tb[col] = tb[col].astype('category')

    tb['weights'] = tb['weights'].astype(float)
    tb['subclass'] = tb['subclass'].astype(int)

    tb = tb[[id, 'distance', 'weights','subclass']]

    return(tb.merge(df, on=id, how='left'))
