from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
ROOT=Path(__file__).resolve().parent
# Concatenation preserves the ordering within the original experiment's splits.
parts=[pd.read_csv(ROOT/f'{name}.csv') for name in ['train','validation','test']]
d=pd.concat(parts,ignore_index=True)
train=np.arange(len(parts[0]));valid=np.arange(len(parts[0]),len(parts[0])+len(parts[1]));test=np.arange(len(parts[0])+len(parts[1]),len(d))
features=['RIDAGEYR','BMXBMI','BMXWAIST']
counts={'complete_eligible_records':len(d)}
model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000))])
w=d.WTPH2YR.to_numpy(); w=w/w[train].mean()
model.fit(d[features].iloc[train],d.target.iloc[train],model__sample_weight=w[train])
pv=model.predict_proba(d[features].iloc[valid])[:,1]; yv=d.target.iloc[valid].to_numpy()
# Educational referral threshold: target weighted validation sensitivity >=80%; not a clinical policy.
thresholds=np.unique(pv)
eligible=[t for t in thresholds if np.sum(w[valid]*(yv==1)*(pv>=t))/np.sum(w[valid]*(yv==1))>=.8]
threshold=float(max(eligible))
pt=model.predict_proba(d[features].iloc[test])[:,1]; yt=d.target.iloc[test].to_numpy(); wt=w[test]
def metrics(y,p,weights):
 tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1],sample_weight=weights).ravel()
 return {'n':len(y),'positives':int(sum(y)), 'roc_auc':float(roc_auc_score(y,p,sample_weight=weights)), 'average_precision':float(average_precision_score(y,p,sample_weight=weights)), 'brier':float(brier_score_loss(y,p,sample_weight=weights)), 'sensitivity':float(tp/(tp+fn)), 'specificity':float(tn/(tn+fp)), 'false_positive_rate':float(fp/(fp+tn)), 'false_negative_rate':float(fn/(fn+tp))}
groups=[]
for field in ['RIAGENDR','RIDRETH3']:
 for value in sorted(d[field].dropna().unique()):
  mask=d[field].iloc[test].to_numpy()==value
  if len(np.unique(yt[mask]))==2:groups.append({'field':field,'code':float(value),**metrics(yt[mask],pt[mask],wt[mask])})
for name,lo,hi in [('18–39',18,40),('40–59',40,60),('60+',60,100)]:
 mask=d.RIDAGEYR.iloc[test].between(lo,hi,inclusive='left').to_numpy()
 if len(np.unique(yt[mask]))==2:groups.append({'field':'age','group':name,**metrics(yt[mask],pt[mask],wt[mask])})
results={'cohort':counts,'split':{'train':len(train),'validation':len(valid),'test':len(test)},'target':'Measured HbA1c >=5.7%, not diagnosis or future onset','threshold':threshold,'threshold_selection':'Highest validation threshold meeting weighted sensitivity >=0.80; exploratory only','test_weighted':metrics(yt,pt,wt),'test_unweighted':metrics(yt,pt,np.ones(len(test))),'baseline_brier_weighted':float(brier_score_loss(yt,np.full(len(yt),np.average(d.target.iloc[train],weights=w[train])),sample_weight=wt)),'groups':groups}
(ROOT/'evaluation.json').write_text(json.dumps(results,indent=2))
scale=model.named_steps['scale']; lr=model.named_steps['model']
spec={'features':features,'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'coefficients':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'referral_threshold':threshold,'training_ranges':{f:[float(d[f].iloc[train].min()),float(d[f].iloc[train].max())] for f in features},'outcome':results['target'],'version':'educational-baseline-1'}
(ROOT/'model.json').write_text(json.dumps(spec,indent=2))
demo=d.iloc[test[:30]][features+['LBXGH','target']].copy();demo['prediction']=pt[:30];demo.to_csv(ROOT/'demo_cases.csv',index=False)

print(json.dumps(results['test_weighted'],indent=2))
