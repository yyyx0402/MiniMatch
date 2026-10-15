import numpy as np

GAINS = np.array([0.0,0.01,0.1,1.0])


def ndcg(labels, scores, product_ids, k=10):
    labels, scores = np.asarray(labels,dtype=int),np.asarray(scores,dtype=float)
    if not len(labels):
        return 0.0
    # Stable product-id tie breaking is shared by every model and the app.
    order = np.lexsort((np.asarray(product_ids,dtype=str),-scores))[:k]
    gains=GAINS[labels]
    discount=1/np.log2(np.arange(min(k,len(labels)))+2)
    ideal=np.dot(np.sort(gains)[::-1][:k],discount)
    return float(np.dot(gains[order],discount)/ideal) if ideal>0 else 0.0


def evaluate(frame, predictions):
    groups=list(frame.groupby("query_group",sort=False))
    result={}
    per_query=[]
    for name,scores in predictions.items():
        values=[]
        reciprocal=[]
        for key,group in groups:
            indexes=group.index.to_numpy()
            value=ndcg(group.label.to_numpy(),np.asarray(scores)[indexes],group.product_id.to_numpy())
            order=np.lexsort((group.product_id.to_numpy(),-np.asarray(scores)[indexes]))
            exact=np.flatnonzero(group.label.to_numpy()[order]==3)
            values.append(value)
            reciprocal.append(1/(exact[0]+1) if len(exact) else 0)
            per_query.append({"query_group":key,"query":str(group.iloc[0]["query"]),"model":name,"ndcg_at_10":value})
        result[name]={"ndcg_at_10":float(np.mean(values)),"mrr_exact":float(np.mean(reciprocal)),"queries":len(values)}
    return result,per_query


def paired_bootstrap(per_query, left="lambdamart",right="bm25",seed=42):
    import pandas as pd
    table=pd.DataFrame(per_query).pivot(index="query_group",columns="model",values="ndcg_at_10")
    delta=(table[left]-table[right]).to_numpy()
    rng=np.random.default_rng(seed)
    bootstrap=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
    return {"comparison":f"{left} - {right}","mean_delta":float(delta.mean()),
            "ci95":[float(x) for x in np.quantile(bootstrap,[.025,.975])],
            "queries":len(delta),"resamples":2000,"seed":seed}
