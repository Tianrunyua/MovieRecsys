from torch.autograd import no_grad



def test_eval(early_stop,epoch, model, data, prefix, ranklist,result_log_path):
    print(prefix + ' start...')
    model.eval()

    with no_grad():
        precision_50, recall_50, ndcg_score_50, \
        precision_20, recall_20, ndcg_score_20, \
        precision_10, recall_10, ndcg_score_10, \
        precision_5, recall_5, ndcg_score_5, \
        precision_1, recall_1, ndcg_score_1 = model.full_accuracy(data, ranklist)
        print(
            '---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
                epoch, precision_10, recall_10, ndcg_score_10))
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------' + prefix + ': early_stop:{0}---------------------------------'.format(early_stop))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------' + prefix + ': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 50, precision_50, recall_50, ndcg_score_50))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------' + prefix + ': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 20, precision_20, recall_20, ndcg_score_20))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------' + prefix + ': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 10, precision_10, recall_10, ndcg_score_10))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------' + prefix + ': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 5, precision_5, recall_5, ndcg_score_5))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------' + prefix + ': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 1, precision_1, recall_1, ndcg_score_1))  # 将字符串写入文件中
            f.write("\n")
        return precision_10, recall_10, ndcg_score_10


def train_eval(epoch, model, prefix, ranklist, result_log_path):
    print(prefix + ' start...')
    model.eval()

    with no_grad():
        precision_50, recall_50, ndcg_score_50, \
        precision_20, recall_20, ndcg_score_20, \
        precision_10, recall_10, ndcg_score_10, \
        precision_5, recall_5, ndcg_score_5, \
        precision_1, recall_1, ndcg_score_1 = model.accuracy(ranklist)
        print(
            '---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
                epoch, precision_10, recall_10, ndcg_score_10))
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 50, precision_50, recall_50, ndcg_score_50))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 20, precision_20, recall_20, ndcg_score_20))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 10, precision_10, recall_10, ndcg_score_10))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 5, precision_5, recall_5, ndcg_score_5))  # 将字符串写入文件中
            f.write("\n")
        with open(result_log_path, "a") as f:
            f.write(
                '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                    epoch, 1, precision_1, recall_1, ndcg_score_1))  # 将字符串写入文件中
            f.write("\n")
    return precision_10, recall_10, ndcg_score_10