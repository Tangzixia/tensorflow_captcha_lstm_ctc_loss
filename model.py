#coding=utf-8
import tensorflow as tf
import numpy as np
import sys
from load_data_sparse import random_batch

reload(sys)
sys.setdefaultencoding('utf8')

def my_model(x_input,targets,seq_length,num_hidden,num_classes=63):
    shape_=x_input.get_shape().as_list()
    #input_shape:[batch_size,170,80]
    x_input=tf.transpose(x_input,(1,0,2))
    cell=tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
    stack=tf.contrib.rnn.MultiRNNCell([cell]*2,state_is_tuple=True)
    outputs,_=tf.nn.dynamic_rnn(cell,x_input,seq_length,dtype=tf.float32,time_major=True)
    outputs=tf.reshape(outputs,[-1,num_hidden])
    fc_weight=tf.get_variable(name="fc_weight",shape=[num_hidden,num_classes],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc_bias=tf.get_variable(name="fc_bias",shape=[num_classes],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
    fc=tf.matmul(outputs,fc_weight)+fc_bias
    fc=tf.reshape(fc,[-1,shape_[1],num_classes])
    fc=tf.transpose(fc,(1,0,2))
    #output_shape:[seq_length,batch_size,num_classes]

    loss=tf.reduce_mean(tf.nn.ctc_loss(labels=targets,inputs=fc,sequence_length=seq_length))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    decoded,log_prob=tf.nn.ctc_beam_search_decoder(fc,seq_length,merge_repeated=False)
    acc=tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0],tf.int32),targets))
    initial=tf.global_variables_initializer()
    return optimizer,initial,acc,loss

if __name__=="__main__":
    x_input=tf.placeholder(tf.float32,[None,170,80])
    # values_=tf.placeholder(dtype=tf.int32,shape=[None,])
    # indices_=tf.placeholder(dtype=tf.int64,shape=[None,63])
    # shape_ = tf.placeholder(dtype=tf.int64, shape=[180])
    # targets=tf.SparseTensor(values=values_,indices=indices_,dense_shape=[62]*63)
    targets = tf.sparse_placeholder(tf.int32)
    seq_length=tf.placeholder(tf.int32,[None])
    optimizer,initial,acc,loss=my_model(x_input,targets,seq_length,1000)
    dir="/home/jobs/Desktop/code/captcha/training"
    batches_img_initial, batches_indices_initial, batches_values_initial, batches_dense_indices_initial = random_batch(20,dir)
    batches_img_test,batches_indices_test,batches_values_test,batches_dense_indices_test=random_batch(20)
    with tf.Session() as sess:
        sess.run(initial)
        for i in range(20000):
            batches_img,batches_indices,batches_values,batches_dense_indices=batches_img_initial,batches_indices_initial,batches_values_initial,batches_dense_indices_initial
            #random.shuffle(batches_img)
            #random.shuffle(batches_indices)
            #random.shuffle(batches_values)
            #random.shuffle(batches_dense_indices)

            per_acc=0.0
            per_loss=0.0
            print("第"+str(i)+"次训练中。。。")
            for i_ in range(len(batches_indices_initial)):
                _,train_acc,train_loss=sess.run([optimizer,acc,loss],feed_dict={x_input:batches_img_initial[i_],targets:(batches_indices_initial[i_],batches_values_initial[i_],batches_dense_indices_initial[i_]),seq_length:np.ones(20)*170})
                per_acc+=train_acc
                per_loss+=train_loss
            print("第"+str(i)+"次epoch的准确率："+str(per_acc/len(batches_img_initial))+",损失为："+str(per_loss/len(batches_img_initial)))
            print("-----测试集上的输出-----\n")
            per_acc_test=0.0
            per_loss_test=0.0
            for i_ in range(len(batches_img_test)):
                #targets:tf.SparseTensorValue(batches_label_test[i_])
                test_acc,test_loss=sess.run([acc,loss],feed_dict={x_input:batches_img_test[i_],targets:(batches_indices_test[i_],batches_values_test[i_],batches_dense_indices_test[i_]),seq_length:np.ones(20)*170})
                per_acc_test+=test_acc
                per_loss_test+=test_loss
            print("第" + str(i) + "次测试集上的准确率：" + str(per_acc_test / len(batches_img_test)) + ",损失为：" + str(per_loss_test / len(batches_img_test)))








