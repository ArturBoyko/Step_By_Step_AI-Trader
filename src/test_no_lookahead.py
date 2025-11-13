# src/test_no_lookahead.py
import numpy as np, pandas as pd
from features_engine import prepare_features
from utils import drop_warmup

def test_no_lookahead():
    np.random.seed(7)
    T=500
    close=np.cumsum(np.random.normal(0,0.0006,T))+1.08
    open_=np.r_[close[0]-0.0003,close[:-1]]
    high=np.maximum(open_,close)+np.abs(np.random.normal(0.0002,0.0001,T))
    low=np.minimum(open_,close)-np.abs(np.random.normal(0.0002,0.0001,T))
    vol=np.random.lognormal(11,0.35,T)
    raw=pd.DataFrame({'open':open_,'high':high,'low':low,'close':close,'volume':vol})
    full=drop_warmup(prepare_features(raw))
    idxs=list(range(150,400,50))
    for rid in idxs:
        pref=drop_warmup(prepare_features(raw.iloc[:rid+1].copy()))
        if len(pref)==0 or rid>=len(full): 
            continue
        a = full.iloc[rid]
        b = pref.iloc[-1]
        common = [c for c in full.columns if c in pref.columns]
        for c in common:
            av, bv = a[c], b[c]
            if not (pd.isna(av) or pd.isna(bv)):
                assert np.isclose(av,bv,rtol=1e-9,atol=1e-12), f"Leak at rid={rid}, col={c}: {av} vs {bv}"

if __name__ == "__main__":
    test_no_lookahead()
