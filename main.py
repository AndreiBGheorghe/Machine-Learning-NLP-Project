import os, re, string, pandas as pd, numpy as np, torch
from wordfreq import zipf_frequency
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset

RO_BERT_MODEL = "readerbench/RoBERT-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

def load_and_clean(path):
    df = pd.read_csv(
        path,
        encoding='utf-8', sep=',', engine='python', quotechar='"', skip_blank_lines=True, on_bad_lines='warn'
    )
    df = df[df["token"].notna() & (df["token"].str.strip() != "")]
    return df.reset_index(drop=True)

train_df = load_and_clean("date/train.csv")
test_df  = load_and_clean("date/test.csv")
nlp = spacy.load("ro_core_news_lg")

def count_syllables(word):
    vowels = "aeiouăîâAEIOUĂÎÂ"
    count, prev = 0, False
    for c in word:
        if c in vowels:
            if not prev:
                count += 1
                prev = True
        else:
            prev = False
    return count

def extract_traditional_features(row):
    sent, tok = row["sentence"], row["token"]
    doc = nlp(sent)
    match = next((t for t in doc if t.text.lower() == tok.lower()), None)
    if match:
        vec = match.vector; pos = match.pos_
        is_ent = int(match.ent_type_ != ""); is_stop = int(match.is_stop)
        dep_ch = len(list(match.children))
    else:
        vec = nlp(tok).vector; pos=None; is_ent=is_stop=dep_ch=0
    freq = zipf_frequency(tok.lower(), "ro"); syl = count_syllables(tok)
    vow = sum(c.lower() in "aeiouăîâ" for c in tok); cons = len(tok) - vow
    sent_len = len(doc); avg_len = np.mean([len(w) for w in sent.split()])
    noun_count = sum(1 for w in doc if w.pos_=="NOUN")
    verb_count = sum(1 for w in doc if w.pos_=="VERB")
    syll_ratio = syl / max(1, len(tok))
    feats = {f"v{i}": vec[i] for i in range(vec.shape[0])}
    feats.update({
        "freq":freq, "len":len(tok), "syllables":syl,
        "vowel_count":vow, "cons_count":cons, "syllable_ratio":syll_ratio,
        "avg_word_len":avg_len, "sentence_length":sent_len,
        "noun_count":noun_count, "verb_count":verb_count,
        "has_digit":int(any(c.isdigit() for c in tok)),
        "is_punct":int(all(c in string.punctuation for c in tok)),
        "is_upper":int(tok.isupper()), "is_title":int(tok.istitle()),
        "is_stop":is_stop, "is_entity":is_ent, "dep_children":dep_ch,
        **{f"pos_{p}":int(pos==p) for p in ["ADJ","NOUN","VERB","ADV","PROPN","PRON"]}
    })
    return feats

def build_traditional_X(df):
    feats = df.apply(extract_traditional_features, axis=1)
    return pd.DataFrame(list(feats.values))

tokenizer = AutoTokenizer.from_pretrained(RO_BERT_MODEL)
bert_model = AutoModel.from_pretrained(RO_BERT_MODEL).to(DEVICE)
bert_model.eval()

def mean_pooling(out, mask):
    emb = out.last_hidden_state
    m = mask.unsqueeze(-1).expand(emb.size()).float()
    summed = (emb * m).sum(1); counts = m.sum(1).clamp(min=1e-9)
    return summed / counts

def extract_bert_embeddings(sentences):
    all_emb=[]
    for i in range(0,len(sentences),BATCH_SIZE):
        batch = sentences[i:i+BATCH_SIZE]
        enc = tokenizer(batch,padding=True,truncation=True,return_tensors="pt").to(DEVICE)
        with torch.no_grad(): out = bert_model(**enc)
        pooled = mean_pooling(out, enc["attention_mask"]).cpu().numpy()
        all_emb.append(pooled)
    return np.vstack(all_emb)

X_trad = build_traditional_X(train_df)
X_bert = pd.DataFrame(extract_bert_embeddings(train_df["sentence"].tolist()), columns=[f"b{i}" for i in range(768)])
X_full = pd.concat([X_trad.reset_index(drop=True), X_bert.reset_index(drop=True)], axis=1)
y = train_df["score"].astype(float).values
X_train,X_val,y_train,y_val = train_test_split(X_full,y,test_size=0.2,random_state=42)

param_grid={"model__n_estimators":[100,200,300],"model__max_depth":[3,5,7],"model__learning_rate":[0.01,0.05,0.1]}
gb_pipe=Pipeline([("scaler",StandardScaler()),("model",GradientBoostingRegressor())])
gs=GridSearchCV(gb_pipe,param_grid,cv=3,scoring="r2",n_jobs=-1,verbose=1)
gs.fit(X_train,y_train)
best_gb=gs.best_estimator_
print("Best GB params:",gs.best_params_)

models={"Ridge":Ridge(),"SVR":SVR(),"LightGBM":LGBMRegressor(),"XGBoost":XGBRegressor(objective="reg:squarederror"),"CatBoost":CatBoostRegressor(verbose=0),"GB_Optimized":best_gb}
results={}
for n,m in models.items():
    p=Pipeline([("scaler",StandardScaler()),("model",m)])
    p.fit(X_train,y_train); preds=p.predict(X_val)
    results[n]=r2_score(y_val,preds); print(f"{n}: R²={results[n]:.4f}")

hf_ds=Dataset.from_pandas(train_df[["sentence","score"]].rename(columns={"sentence":"text","score":"label"}))
def tok_fn(ex): return tokenizer(ex["text"],truncation=True)
hf_ds=hf_ds.map(tok_fn,batched=True).train_test_split(test_size=0.2,seed=42)

model_reg=AutoModelForSequenceClassification.from_pretrained(RO_BERT_MODEL,num_labels=1,problem_type="regression").to(DEVICE)
args=TrainingArguments(
    output_dir="bert-finetuned",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
)

data_collator=DataCollatorWithPadding(tokenizer)
def compute_r2(ep): p,l=ep; return {"r2":r2_score(l,p.squeeze())}
trainer=Trainer(model=model_reg,args=args,train_dataset=hf_ds["train"],eval_dataset=hf_ds["test"],tokenizer=tokenizer,data_collator=data_collator,compute_metrics=compute_r2)
trainer.train(); finetuned_metrics=trainer.evaluate()
print("BERT R²:",finetuned_metrics['eval_r2'])

best_name,best_score=max(results.items(),key=lambda x:x[1])
if finetuned_metrics['eval_r2']>best_score: best_name='RoBERT-finetuned'
best_model=models.get(best_name,best_gb)
print(f"Best: {best_name} (R²={max(best_score,finetuned_metrics['eval_r2']):.4f})")
X_trad_test=build_traditional_X(test_df)
X_bert_test=pd.DataFrame(extract_bert_embeddings(test_df["sentence"].tolist()),columns=[f"b{i}" for i in range(768)])
X_test_full=pd.concat([X_trad_test,X_bert_test],axis=1)

pipe=Pipeline([("scaler",StandardScaler()),("model",best_model)])
pipe.fit(X_full,y); p1=pipe.predict(X_test_full)
enc=tokenizer(test_df["sentence"].tolist(),padding=True,truncation=True,return_tensors="pt").to(DEVICE)
with torch.no_grad(): out=model_reg(**enc)
p2=out.logits.cpu().squeeze().numpy()
preds=np.clip((p1+p2)/2,0,1)
sub=pd.DataFrame({"Id":test_df["Id"],"score":preds}); sub.to_csv("submission.csv",index=False)
print("saved submission")