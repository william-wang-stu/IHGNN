# Data Preparation
 
We put all datasets under a sub-directory `./data`. For ease management, we suggest organize datasets like this

```
$DATA/
|–– Twitter-Huangxin/
|–– Weibo-Aminer/
```

Datasets list:
- [Twitter-Huangxin](#twitter)
- [Weibo-Aminer](#weibo)

The instructions to preprocess each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets.

### Twitter-Huangxin

The sub-directory file structure looks like this
```
Twitter-Huangxin/
|–– edges.data/
|–– cascades.data/ # key-value map
|–– train.data/
|–– valid.data/
|–– test.data/
|–– u2idx.data/
|–– idx2u.data/
```

In general, file `cascades.data` contains 29,192 tag-based cascades, and file `edges.data` contains 10,269 users on Twitter.

Specifically, file `edges.data` contains an array of directional edges in the social network, i.e. `[3092,4076]`. File `cascades.data` contains a series of key-value pairs. An illustrative cascade example is given below,

```
#antifadomesticterrorists: {
    user: [32839, 16621, 44225, 421, 44397, 26952, 12685, 23522, 17801, 44134],
    ts: [1562135807, 1562495385, 1564994067, 1566269187, 1568032965, 1568321964, 1569850925, 1571147642, 1571572412, 1572695879],
    content: ["RT @PoultryPpa: They are rich in high-quality protein, healthy fats and many essential vitamins and minerals.", ...]
}
```

As for files `train/valid/test.data`, we randomly sample 80\% of `cascades.data` for training, 10\% for validation, and the rest 10\% for testing.
Besides, we anonymize the users and uniquely label each user with a consecutive number, the mappings are stored in files `u2idx/idx2u.data`.

### Weibo-Aminer

The file structure of the sub-directory looks the same as Twitter dataset.
You can pubicly download the Weibo-Aminer dataset from this [link](https://www.aminer.cn/influencelocality).
