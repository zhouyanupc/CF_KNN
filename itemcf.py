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
