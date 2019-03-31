import tensorflow as tf
import DataGen
import os
import time
import sys
import json
import logging
import datetime
class _decoder:
    def __init__(self,unit,h_i,vsize,name):

        self.NUM_UNIT = unit
        self.VEC_SIZE = vsize
        self.H_I_SIZE = h_i.shape[1]
        with tf.name_scope('k',"decoder_"+name) as scope:
            self.C_r = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.H_I_SIZE],name=name+"c_r")
            self.C_z = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.H_I_SIZE],name=name+"c_z")
            self.C_h = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.H_I_SIZE],name=name+"c_h")
            self.U_r = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name=name+"u_r")
            self.U_z = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name=name+"u_z")
            self.U_h = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name=name+"u_h")
            self.W_r = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.VEC_SIZE],name=name+"w_r")
            self.W_z = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.VEC_SIZE],name=name+"w_z")
            self.W_h = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.VEC_SIZE],name=name+"w_h")
            self.ch_r = tf.matmul(h_i,self.C_r,transpose_b=True)
            self.ch_z = tf.matmul(h_i,self.C_z,transpose_b=True)
            self.ch_h = tf.matmul(h_i,self.C_h,transpose_b=True)
    def call(self,state,x):
        r = tf.nn.sigmoid(tf.matmul(x,self.W_r,transpose_b=True)+tf.matmul(state,self.U_r,transpose_b=True)+self.ch_r)
        pass_state = r*state
        z = tf.nn.sigmoid(tf.matmul(x,self.W_z,transpose_b=True)+tf.matmul(state,self.U_z,transpose_b=True)+self.ch_z)
        h = tf.nn.tanh(tf.matmul(x,self.W_h,transpose_b=True)+tf.matmul(pass_state,self.U_h,transpose_b=True)+self.ch_h)
        h_t = (1-z)*state + z*h
        return h_t

    def dynamic_run(self,inputs,seq_len,batch_size,max_len):
        mlen = tf.reduce_max(seq_len)
        i = tf.constant(0)
        zero_state = tf.constant(value=0 ,dtype=tf.float32,shape=[batch_size,self.NUM_UNIT])
        h_ta = tf.TensorArray(dtype=tf.float32,size=max_len)
        def time_step(time,inputs,state,ta):
            x_i = inputs[:,time,:]
            h_i = self.call(state,x_i)
            ta.write(time,h_i)
            time += 1
            return time,inputs,h_i,ta
        i,_,h,h_ta = tf.while_loop(
            cond = lambda time,*_: time < mlen,
            body = time_step,
            loop_vars=[i,inputs,zero_state,h_ta]
        )
        res = h_ta.stack()


        return tf.reshape(res,[64,max_len,self.NUM_UNIT])

class skip_thought:
    def __init__(self):
        self.NUM_UNIT = 300

        self.NUM_UNIT_DE = self.NUM_UNIT
        self.WORD_VEC = 300
        self.BATCH_SIZE = 64
        self.VEC_SIZE = 170000
        self.MAX_LENGTH = 50
        self.LR = 0.0001

    def get_cell(self):
        # gru_cell = tf.keras.layers.GRUCell(self.NUM_UNIT)

        gru_cell = tf.nn.rnn_cell.GRUCell(self.NUM_UNIT)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.NUM_UNIT)

        return gru_cell



    def model(self):
        sen_i = tf.placeholder(tf.int32,[self.BATCH_SIZE,self.MAX_LENGTH],name="INPUT_SEN")
        sen_i_pre = tf.placeholder(tf.int32,[self.BATCH_SIZE,self.MAX_LENGTH],name="INPUT_SEN")
        sen_i_post = tf.placeholder(tf.int32,[self.BATCH_SIZE,self.MAX_LENGTH],name="INPUT_SEN")

        length_i = tf.placeholder(tf.int32,[self.BATCH_SIZE])
        length_pre = tf.placeholder(tf.int32,[self.BATCH_SIZE])
        length_post = tf.placeholder(tf.int32,[self.BATCH_SIZE])

        embedding_v = tf.get_variable("embedding_lookup",[self.VEC_SIZE,self.WORD_VEC],dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer())
        sen_i_emb = tf.nn.embedding_lookup(embedding_v,sen_i)
        sen_pre_emb = tf.nn.embedding_lookup(embedding_v,sen_i_pre)
        sen_post_emb = tf.nn.embedding_lookup(embedding_v,sen_i_post)

        encoder = self.get_cell()

        init_state = encoder.zero_state(self.BATCH_SIZE,dtype=tf.float32)
        _,h_i = tf.nn.dynamic_rnn(encoder,sen_i_emb,length_i,initial_state=init_state)

        decoder_pre  = self.get_cell()
        decoder_post = self.get_cell()

        pre_h,outpre = tf.nn.dynamic_rnn(decoder_pre,sen_pre_emb,length_pre,initial_state=h_i,scope="PRE")
        post_h,outpre = tf.nn.dynamic_rnn(decoder_post,sen_post_emb,length_post,initial_state=h_i,scope="POST")

        # decoder_pre = _decoder(self.NUM_UNIT_DE,h_i,self.WORD_VEC,"PRE")
        # decoder_post = _decoder(self.NUM_UNIT_DE,h_i,self.WORD_VEC,"POST")
        #
        # pre_h = decoder_pre.dynamic_run(inputs=sen_pre_emb,seq_len=length_pre,batch_size=self.BATCH_SIZE,max_len=self.MAX_LENGTH)
        # post_h = decoder_post.dynamic_run(inputs=sen_post_emb,seq_len=length_post,batch_size=self.BATCH_SIZE,max_len=self.MAX_LENGTH)

        w_out = tf.get_variable(shape=[self.NUM_UNIT_DE,self.VEC_SIZE],dtype=tf.float32,initializer=tf.glorot_normal_initializer(),name="w_out")

        out_pre = tf.tensordot(pre_h,w_out,axes=[2,0])
        out_post = tf.tensordot(post_h,w_out,axes=[2,0])


        label_pre = tf.one_hot(sen_i_pre,depth=self.VEC_SIZE,axis=-1)
        label_post = tf.one_hot(sen_i_pre,depth=self.VEC_SIZE,axis=-1)

        result_pre = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_pre,labels=label_pre)
        result_post = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_post,labels=label_post)

        loss = tf.reduce_mean(result_pre)+tf.reduce_mean(result_post)

        tf.summary.scalar("Loss",loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.LR)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)

        grads = optimizer.compute_gradients(loss)
        # print(grads)
        for i,(grad,v) in enumerate(grads):

            tf.summary.histogram(v.name+'/gradient',grad)

        grad_v,var = zip(*grads)
        grad_v,global_norm = tf.clip_by_global_norm(grad_v,5)
        grads = zip(grad_v,var)
        train = optimizer.apply_gradients(grads)
        merge = tf.summary.merge_all()



        opt = {
            'sen_i':sen_i,
            'sen_pre':sen_i_pre,
            'sen_post':sen_i_post,
            'length_i':length_i,
            'length_pre':length_pre,
            'length_post':length_post,
            'loss':loss,
            'train':train,
            'merge':merge,

        }
        return opt

    def run_model(self,meta):
        epoch = 50
        source_name = meta['train_data'] # meta
        checkpoint_dir = os.path.abspath(meta['checkpoint_dir']) # meta
        summary_dir = os.path.abspath(meta['summary_dir']) # meta
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        p = DataGen.Preprocessor(SEG_BY_WORD=meta['seg_by_word'])
        data_meta = meta['data_meta'] # meta

        ops = self.model()

        # 训练过程
        saver = tf.train.Saver()
        config = tf.ConfigProto(
            # log_device_placement=True

        )
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            sess.graph.finalize()
            train_writer = tf.summary.FileWriter(summary_dir,sess.graph)
            start_epoch = 0
            global_step = 0
            if checkpoint:
                saver.restore(sess, checkpoint)
                print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
                start_epoch += int(checkpoint.split('-')[-1])
                global_step += int(checkpoint.split('-')[-2])
            start_time = time.time()

            # 开始训练
            for i in range(start_epoch,epoch):
                data_gen = p.data_provider(source_name, meta=data_meta)
                try:
                    batch_count = 0
                    while True:
                        try:
                            last_time = time.time()

                            batch_data = next(data_gen)


                            _,loss,merge = sess.run([ops['train'],ops['loss'],ops['merge']],
                                                    feed_dict={
                                ops['sen_i']:batch_data[0],
                                ops['sen_pre']:batch_data[1],
                                ops['sen_post']:batch_data[2],
                                ops['length_i']:batch_data[3],
                                ops['length_pre']:batch_data[4],
                                ops['length_post']:batch_data[5],
                            })

                            cur_time =time.time()
                            time_cost = cur_time-last_time
                            total_cost = cur_time-start_time
                            if global_step % 1 == 0:
                                train_writer.add_summary(merge,global_step)
                                # logger.write_log([global_step/10,loss,total_cost])
                            print('[INFO] Batch %d 训练结果：LOSS=%.2f  用时: %.2f 共计用时 %.2f' % (batch_count, loss,time_cost,total_cost))

                            # print('[INFO] Batch %d'%batch_count)
                            # matplotlib 实现可视化loss
                            batch_count += 1
                            global_step += 1
                        except StopIteration:
                            print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                            saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)
                            break
                        except Exception as e:
                            logging.exception(e)
                            print("[INFO] 因为程序错误停止训练，开始保存模型")
                            saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)
                except StopIteration:
                    print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                    saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)

                except KeyboardInterrupt:
                    print("[INFO] 强行停止训练，开始保存模型")
                    saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)
                    break

meta={
    'name':'SKIP_T',
    'seg_by_word': True,
    'train_data':'DOC_SEG.txt',
    'checkpoint_dir':'checkpoint_SKT',
    'summary_dir':'summary_SKT',
    'data_meta':{
        "NAME":"SKIP_THOUGHT",
        "LEN":50,
        "BATCH":64,
    }
}
st = skip_thought()
st.run_model(meta)