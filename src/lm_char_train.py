# src/lm_char_train.py
import argparse, h5py, os, torch, torch.nn as nn

def gather_texts(root_or_file, splits=("train","val")):
    paths=[]
    if os.path.isdir(root_or_file):
        for d in os.listdir(root_or_file):
            p=os.path.join(root_or_file,d)
            for s in splits:
                f=os.path.join(p,f"data_{s}.hdf5")
                if os.path.isfile(f): paths.append(f)
    else:
        paths=[root_or_file]
    texts=[]
    for p in paths:
        with h5py.File(p,"r") as f:
            if "sentences" in f:
                texts += [s.decode().lower() for s in f["sentences"][:]]
    return texts

class CharLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.stoi = {c:i for i,c in enumerate(vocab)}
        self.emb = nn.Embedding(len(vocab), 64)
        self.rnn = nn.GRU(64, 256, batch_first=True)
        self.fc  = nn.Linear(256, len(vocab))
    def forward(self, x, h=None):
        z = self.emb(x); z, h = self.rnn(z, h); return self.fc(z), h

def batchify(texts, stoi, T=256, B=64):
    import random
    corpus = "\n".join(texts)
    toks = [stoi.get(c, stoi["<unk>"]) for c in corpus]
    for i in range(0, max(0,len(toks)-T-1), T):
        seq = toks[i:i+T+1]
        if len(seq)<T+1: break
        x = torch.tensor(seq[:-1]).view(1,T)
        y = torch.tensor(seq[1:]).view(1,T)
        yield x.repeat(B,1), y.repeat(B,1)

def main(a):
    # char vocab: letters, digits, space, basic punct, \n, and <unk>
    base = "abcdefghijklmnopqrstuvwxyz0123456789 '"
    punct = ",.;:?!"
    vocab = ["<unk>"] + list(base + punct + "\n")
    texts = gather_texts(a.data, splits=("train","val"))
    os.makedirs("checkpoints", exist_ok=True)
    lm = CharLM(vocab); opt = torch.optim.Adam(lm.parameters(), lr=2e-3)
    lossfn = nn.CrossEntropyLoss()
    for epoch in range(a.epochs):
        total=0; steps=0
        for X,Y in batchify(texts, {c:i for i,c in enumerate(vocab)}, T=a.seq_len, B=a.batch):
            logits,_ = lm(X)
            loss = lossfn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); steps+=1
            if steps % 50 == 0: print(f"[lm] epoch {epoch+1} step {steps} loss {total/steps:.3f}")
    torch.save({"state": lm.state_dict(), "vocab": vocab}, "checkpoints/lm_char.pt")
    print("saved checkpoints/lm_char.pt")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="BT25 root or a single h5 file")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    main(p.parse_args())

