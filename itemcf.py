from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from surprise import accuracy

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
train_set = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# K折交叉验证,k=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions,verbose=True)

uid = str(196)
iid = str(302)

pred = algo.predict(uid, iid)
print(pred)

"""
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8596
MAE:  0.6570
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8550
MAE:  0.6539
Computing the msd similarity matrix...
Done computing similarity matrix.
RMSE: 0.8549
MAE:  0.6541
user: 196        item: 302        r_ui = None   est = 3.98   {'actual_k': 50, 'was_impossible': False}
"""
