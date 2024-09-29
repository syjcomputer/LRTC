from __future__ import print_function

from datetime import datetime
import torch.backends.cudnn as cudnn

from RiskAnalysis.utils import *
from RiskAnalysis.data_process import risk_dataset
from RiskAnalysis.risker import risk_torch_model
import RiskAnalysis.risker.risk_torch_model as risk_model
from RiskAnalysis.common import config as config_risk
from copy import deepcopy

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
import keras
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, Lambda
import tensorflow as tf


def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]

        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                right += 1
        total += len(y_true)
    return right / total

class Evaluator(keras.callbacks.Callback):
    def __init__(self, vaild_generator, model):
        self.best_val_acc = 0.
        self.vaild_generator = vaild_generator
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.vaild_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights( args['store_name']+ 'best_model.weights')
            # self.model.save(cur_path + 'best_model.h5')
        print(val_acc)
        print(self.best_val_acc)

def output_risk_scores(file_path, id_2_scores, label_index, ground_truth_y, predict_y):
    op_file = open(file_path, 'w', 1, encoding='utf-8')
    for i in range(len(id_2_scores)):
        _id = id_2_scores[i][0]
        _risk = id_2_scores[i][1]
        _label_index = label_index.get(_id)
        _str = "{}, {}, {}, {}".format(ground_truth_y[_label_index],
                                       predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()
    return True

def prepare_data_4_risk_data(data_name, risk_dataset_rate):
    """
    first, generate , include all_info.csv, train.csv, val.csv, test.csv.
    second, use csvs to generate rules. one rule just judge one class
    :return:
    """
    train_data, validation_data, test_data = risk_dataset.load_data(cfg, data_name, risk_dataset_rate)
    return train_data, validation_data, test_data

def prepare_data_4_risk_model(train_data, validation_data, test_data,
                              learn_confidence, learning_rate, bs):
    rm = risk_torch_model.RiskTorchModel(learn_confidence, learning_rate, bs)
    rm.train_data = train_data
    rm.validation_data = validation_data
    rm.test_data = test_data
    return rm


def train(learn_confidence, learning_rate, bs, nb_epoch, class_num, batch_size, store_name, resume=False,
          start_epoch=0, model_path=None, seed=1234, epoches=5, maxlen = 150, device=0, lr2=5e-5):
    save_name = os.path.join('NLP/risk/', store_name, str(seed))
    if (not os.path.exists(save_name)):
        os.makedirs(save_name)

    path = '{}/{}'.format(store_name, config_risk.global_risk_dataset)
    if (not os.path.exists(path)):
        os.makedirs(path)

    # setup output
    exp_dir = save_name
    # if device==0:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
    # else:
    #     device = torch.device("cpu")

    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available() and device==0
    print(f'--------------Use cuda:{use_cuda}---------------------')

    # Model
    if resume:

        bert = build_transformer_model(
            config_path=config_risk.config_path,
            checkpoint_path=config_risk.checkpoint_path,
            with_pool=True,
            return_keras_model=False,
        )

        classify_output = Dropout(rate=0.5, name='final_Dropout')(bert.model.output)
        classify_output = Dense(units=class_num,  # units是输出层维度
                                activation='softmax',
                                name='classify_output',
                                kernel_initializer=bert.initializer
                                )(classify_output)
        model = keras.models.Model(bert.model.input, classify_output)

        print(model_path)
        model.load_weights(model_path)

        print("---------------- load baseline model successfully!------------------------")
    else:
        model = None
    tokenizer = Tokenizer(config_risk.dict_path, do_lower_case=True) # bert4keras
    # tokenizer = MyTokenizer(get_token_dict(dict_path)) # keras_bert

    # model.fit_generator()
    epochs = 1

    max_test_acc = 0
    # lr = [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.00002]
    train_data, val_data, test_data = prepare_data_4_risk_data(store_name, config_risk.global_risk_dataset)
    risk_data = [train_data, val_data, test_data]

    store_path = os.path.join('risk/{}/{}'.format(store_name, config_risk.global_risk_dataset), 'adaTrain')
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        my_risk_model = prepare_data_4_risk_model(risk_data[0], risk_data[1], risk_data[2],
                                                  learn_confidence, learning_rate, bs)
        train_one_pre = torch.empty((0, 1), dtype=torch.float64)
        val_one_pre = torch.empty((0, 1), dtype=torch.float64)
        test_one_pre = torch.empty((0, 1), dtype=torch.float64)

        # model is bert model in zh_get_risk_dataset
        time1 = datetime.now()
        train_pre, train_labels, train_predictions, train_texts, train_acc, train_loss = base_model_test(model, tokenizer, "train", 128)
        time2 = datetime.now()
        print(f"train time:{time2-time1}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~Epoch:{epoch} train_acc: {train_acc}; train loss: {train_loss}")
        # train_acc=0
        val_pre, val_labels, val_predictions, val_texts, val_acc, val_loss = base_model_test(model, tokenizer, "val", 128)
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~Epoch:{epoch} val_acc: {val_acc}; val loss:{val_loss}")
        test_pre, test_labels, test_predictions, test_texts, test_acc, test_loss = base_model_test(model, tokenizer, "test", 128)
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~Epoch:{epoch} test_acc: {test_acc}, test losss: {test_loss}")

        #path = os.path.join('./{}/{}'.format(store_name, config_risk.global_risk_dataset), 'cur_result.txt')
        path = os.path.join(store_path, 'cur_result.txt')
        if os.path.exists(store_path)==False:
            os.makedirs(store_path)

        with open(path, 'a+') as f:
            f.write("epoch " + str(epoch) + ": TEST ACC = " + str(test_acc) + '\n')

        # list2numpy
        train_pre = np.asarray(train_pre)
        val_pre = np.asarray(val_pre)
        test_pre = np.asarray(test_pre)

        # numpy2tensor
        train_pre = torch.from_numpy(train_pre).cuda()
        val_pre = torch.from_numpy(val_pre).cuda()
        test_pre = torch.from_numpy(test_pre).cuda()

        path = exp_dir + '/' + '{}'.format(config_risk.global_risk_dataset)
        if not os.path.exists(path):
            os.mkdir(path)
        if test_acc > max_test_acc:
            max_test_acc = test_acc

            model.save(os.path.join(path, str(epoch) + '_' + str(max_test_acc) + '_best_model.h5'))
            # net.cpu()
            # torch.save(net, exp_dir + '/' + str(epoch) + '_' + str(max_test_acc) +'_model.pth')
            # net.to(device)
        with open(path + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f\n' % (epoch, test_acc))

        if epoch == 0:
            np.save(exp_dir + '/train_softmax.npy', train_pre.cpu().numpy())
            np.save(exp_dir + '/val_softmax.npy', val_pre.cpu().numpy())
            np.save(exp_dir + '/test_softmax.npy', test_pre.cpu().numpy())

        a, _ = torch.max(train_pre, 1)
        b, _ = torch.max(val_pre, 1)
        c, _ = torch.max(test_pre, 1)

        # store_path = os.path.join('./{}/{}'.format(store_name, config_risk.global_risk_dataset), 'diedai')
        if not os.path.exists(store_path):
            os.mkdir(store_path)

        with open(store_path + '/result.txt', 'a') as f:
                f.write(
                    'epoch {} train: acc {:.4f}  val: acc {:.4f}  test: acc {:.4f} \n'.format(
                        epoch, train_acc, val_acc, test_acc))

        train_one_pre = torch.cat((train_one_pre.cuda(), torch.reshape(a, (-1, 1))), dim=0).cpu().numpy()
        val_one_pre = torch.cat((val_one_pre.cuda(), torch.reshape(b, (-1, 1))), dim=0).cpu().numpy()
        test_one_pre = torch.cat((test_one_pre.cuda(), torch.reshape(c, (-1, 1))), dim=0).cpu().numpy()

        train_labels = torch.argmax(train_pre, 1).cpu().numpy()
        out = np.unique(train_labels)
        print(f"labels:{out}")
        # np.save('train_label.npy', train_labels)
        val_labels = torch.argmax(val_pre, 1).cpu().numpy()
        # np.save('val_label', val_labels)
        test_labels = torch.argmax(test_pre, 1).cpu().numpy()
        # np.save('test_label', test_labels)

        # train risk model
        my_risk_model.train(train_one_pre, val_one_pre, test_one_pre, train_pre.cpu().numpy(),
                            val_pre.cpu().numpy(),
                             test_pre.cpu().numpy(), train_labels, val_labels, test_labels, epoch, epoches, store_name, test_loss)
        # np.save(str(epoch) + 'rule_w', np.array(self.my_risk_model.rule_learn_weights))
        # np.save(str(epoch) + 'rule_sigma', np.array(self.my_risk_model.learn_rule_variance))
        # np.save(str(epoch) + 'mac_sigma', np.array(self.my_risk_model.learn_machine_variances))
        # np.save(str(epoch) + 'func', np.array(self.my_risk_model.func_params))
        my_risk_model.predict(test_one_pre, test_pre.cpu().numpy(), epoch)

        test_num = my_risk_model.test_data.data_len
        test_ids = my_risk_model.test_data.data_ids
        test_pred_y = test_labels
        test_true_y = my_risk_model.test_data.true_labels
        risk_scores = my_risk_model.test_data.risk_values

        id_2_label_index = dict()
        id_2_VaR_risk = []
        for i in range(test_num):
            id_2_VaR_risk.append([test_ids[i], risk_scores[i]])
            id_2_label_index[test_ids[i]] = i
        id_2_VaR_risk = sorted(id_2_VaR_risk, key=lambda item: item[1], reverse=True)
        # if epoch == 0:
        output_risk_scores(store_path + f'/risk_score_{epoch}.txt', id_2_VaR_risk, id_2_label_index, test_true_y, test_pred_y)

        id_2_risk = []
        for i in range(test_num):
            test_pred = test_one_pre[i]
            m_label = test_pred_y[i]
            t_label = test_true_y[i]
            if m_label == t_label:
                label_value = 0.0
            else:
                label_value = 1.0
            id_2_risk.append([test_ids[i], 1 - test_pred])
        id_2_risk_desc = sorted(id_2_risk, key=lambda item: item[1], reverse=True)
        # if epoch == 0:
        output_risk_scores(store_path + f'/base_score_{epoch}.txt', id_2_risk_desc, id_2_label_index, test_true_y, test_pred_y)

        budgets = [10, 20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
        risk_correct = [0] * len(budgets)
        base_correct = [0] * len(budgets)
        for i in range(test_num):
            for budget in range(len(budgets)):
                if i < budgets[budget]:
                    pair_id = id_2_VaR_risk[i][0]
                    _index = id_2_label_index.get(pair_id)
                    if test_true_y[_index] != test_pred_y[_index]:
                        risk_correct[budget] += 1
                    pair_id = id_2_risk_desc[i][0]
                    _index = id_2_label_index.get(pair_id)
                    if test_true_y[_index] != test_pred_y[_index]:
                        base_correct[budget] += 1
        print('risk_correct:{}'.format(risk_correct))
        print('base_correct:{}'.format(base_correct))

        risk_loss_criterion = risk_model.RiskLoss(my_risk_model)
        risk_loss_criterion = risk_loss_criterion.cuda()

        rule_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_discrete(), dtype=torch.float64).cuda()
        machine_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_continue(), dtype=torch.float64).cuda()
        rule_activate = torch.tensor(my_risk_model.test_data.get_rule_activation_matrix(), dtype=torch.float64).cuda()
        machine_activate = torch.tensor(my_risk_model.test_data.get_prob_activation_matrix(),
                                        dtype=torch.float64).cuda()
        machine_one = torch.tensor(my_risk_model.test_data.machine_label_2_one, dtype=torch.float64).cuda()
        risk_y = torch.tensor(my_risk_model.test_data.risk_labels, dtype=torch.float64).cuda()

        # risk_mul_y = torch.tensor(self.my_risk_model.test_data.risk_mul_labels).to(device[0])
        # risk_activate = torch.tensor(self.my_risk_model.test_data.risk_activate).to(device[0])
        # machine_mul_probs = torch.tensor(test_pre).to(device[0])

        test_data = deepcopy(test_pre)

        risk_labels = risk_loss_criterion(test_pre,
                                          rule_mus,
                                          machine_mus,
                                          rule_activate,
                                          machine_activate,
                                          machine_one,
                                          test_data, labels=None)

        risk_labels = risk_labels.cpu().numpy()
        print("risk_labels.shape", risk_labels.shape)
        # numpy2list
        risk_labels.tolist()
        print("len(test_texts)", len(test_texts))
        print("len(risk_labels)", len(risk_labels))

        test_risk_data = load_data2(test_texts, risk_labels)
        val_risk_data = load_data2(val_texts, val_labels)

        ############## keras_bert
        # test_generator = Keras_DataGenerator(test_risk_data, tokenizer, batch_size, maxlen)
        # vaild_generator = Keras_DataGenerator(val_risk_data, tokenizer, batch_size, maxlen)

        ################## bert4keras
        test_generator = data_generator(test_risk_data, tokenizer, maxlen, batch_size)
        vaild_generator = data_generator(val_risk_data, tokenizer, maxlen, batch_size)


        # model.add_loss(test_loss*10)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr2),
            metrics=['accuracy'],

        )
        evaluator = Evaluator(vaild_generator, model)
        # adversarial_training(model, 'Embedding-Token', 0.2)

        model.fit_generator(
            test_generator.forfit(),
            steps_per_epoch=len(test_generator),
            epochs=epochs,
            # class_weight = 'auto',
            callbacks=[evaluator]
        )
    print("------------------start final test----------------------------")
    test_pre, test_labels, test_predictions, test_texts, test_acc, _ = base_model_test(model, tokenizer, "test", 128)
    # nni.report_intermediate_result(test_acc)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~test_acc", test_acc)
    with open(store_path + 'cur_result.txt', 'a+') as f:
        f.write("final " + ": TEST ACC = " + str(test_acc) + '\n')


def main(args):
    model_path = args["model_path2"]
    num_cores = args['num_cores']
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                                         inter_op_parallelism_threads=num_cores,
                                         gpu_options=tf.compat.v1.GPUOptions(
                                             visible_device_list="0",  # choose GPU device number
                                             allow_growth=True
                                         ),
                                         allow_soft_placement=True,
                                         device_count={'CPU': 2})
    session = tf.compat.v1.Session(config=config)
    #K.set_session(session)
    tf.compat.v1.keras.backend.set_session(session)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # NYT
    data_selection = args['data_selection']
    deep_learning_selection = args['deep_learning_selection']

    cfg = config_risk.Configuration(data_selection, deep_learning_selection, config_risk.global_risk_dataset)
    # print(config_risk.global_risk_dataset)
    # print(cfg.get_risk_dataset_path())

    """Seed and GPU setting"""
    seed = args['seed']  # 1234 good
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    train(learn_confidence=args['learn_confidence'],
          learning_rate=args['learning_rate'],
          bs=args['bs'],
          nb_epoch=args['nb_epoch'],  # number of epoch
          class_num=args['class_num'],
          batch_size=args['batch_size'],  # batch size
          store_name=args['store_name'],  # folder for output   fudan
          resume=args['resume'],  # resume training from checkpoint
          start_epoch=0,  # the start epoch number when you resume the training
          model_path=model_path,  # the saved model where you want to resume the training
          seed=args['seed'],
          epoches=args['epoches'],
          maxlen=args['maxlen'],
          device =args['device'],
          lr2 = args["lr2"])

if __name__ == '__main__':
    args = cfg.get_params()
    params = vars(args)
    main(params)
