import tensorflow as tf

class _decoder:
    def __init__(self,unit,h_i,vsize,name):

        self.NUM_UNIT = unit
        self.VEC_SIZE = vsize
        with tf.name_scope("decoder_"+name):
            self.C_r = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name="c_r")
            self.C_z = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name="c_z")
            self.C_h = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name="c_h")
            self.U_r = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name="u_r")
            self.U_z = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name="u_z")
            self.U_h = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.NUM_UNIT],name="u_h")
            self.W_r = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.VEC_SIZE],name="w_r")
            self.W_z = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.VEC_SIZE],name="w_r")
            self.W_h = tf.get_variable(dtype = tf.float32,shape=[self.NUM_UNIT,self.VEC_SIZE],name="w_r")
            self.ch_r = self.C_r * h_i
            self.ch_z = self.C_z * h_i
            self.ch_h = self.C_h * h_i
    def call(self,state,x):
        r = tf.nn.sigmoid(tf.matmul(self.W_r,x)+tf.matmul(self.U_r,state)+self.ch_r)
        pass_state = r*state
        z = tf.nn.sigmoid(tf.matmul(self.W_z,x)+tf.matmul(self.U_z,pass_state)+self.ch_z)
        h = tf.nn.tanh(tf.matmul(self.W_h,x)+tf.matmul(self.U_h,state)+self.ch_h)
        h_t = (1-z)*state + z*h
        return h_t

    def dynamic_run(self,inputs,seq_len):
        mlen = tf.maximum(seq_len)
        i = tf.Constant(0)
        batch_size = tf.shape(inputs)[1]
        zero_state = tf.constant(0,tf.float32,[batch_size,self.NUM_UNIT])
        h_ta = tf.TensorArray(dtype=tf.float32,size=mlen,dynamic_size=None)
        def time_step(time,inputs,state,ta):
            x_i = inputs[:,time,:]
            h_i = self.call(state,x_i)
            ta.write(time,h_i)
            return time,inputs,h_i,ta
        _,_,h,h_ta = tf.while_loop(
            cond = lambda time,*_: time < mlen,
            body = time_step,
            loop_vars=[i,inputs,zero_state,h_ta]
        )
        return h_ta.stack()

class skip_thought:
    def __init__(self):
        self.NUM_UNIT = 100

        self.NUM_UNIT_DE = 200
        self.WORD_VEC = 300
        self.BATCH_SIZE = 64
        self.VEC_SIZE = 10000
        self.MAX_LENGTH = 30
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

        init_state = encoder.zero_state(self.BATCH_SIZE)
        _,h_i = tf.nn.dynamic_rnn(encoder,sen_i_emb,length_i,initial_state=init_state)

        decoder_pre = _decoder(self.NUM_UNIT_DE,h_i,self.VEC_SIZE,"PRE")
        decoder_post = _decoder(self.NUM_UNIT_DE,h_i,self.VEC_SIZE,"POST")

        pre_h = decoder_pre.dynamic_run(inputs=sen_pre_emb,seq_len=length_pre)
        post_h = decoder_post.dynamic_run(inputs=sen_post_emb,seq_len=length_post)

        w_out = tf.get_variable(shape=[self.NUM_UNIT_DE,self.VEC_SIZE],dtype=tf.float32,initializer=tf.glorot_normal_initializer(),name="w_out")
        out_pre = tf.matmul(w_out,pre_h)
        out_post = tf.matmul(w_out,post_h)

        label_pre = tf.one_hot(sen_i_pre,depth=self.VEC_SIZE)
        label_post = tf.one_hot(sen_i_pre,depth=self.VEC_SIZE)

        result_pre = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_pre,labels=label_pre)
        result_post = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_post,labels=label_post)

        loss = tf.reduce_sum(result_pre)+tf.reduce_sum(result_post)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.LR)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)

        grads = optimizer.compute_gradients(loss)

        for i,(grad,v) in enumerate(grads):
            tf.summary.histogram(v.name+'/gradient',grad)

        grad_v,var = zip(*grads)
        grad_v = tf.clip_by_global_norm(grad_v,5)
        grads = zip(grad_v,var)
        train = optimizer.apply_gradients(grads)



        opt = {
            'sen_i':sen_i,
            'sen_pre':sen_i_pre,
            'sen_post':sen_i_post,
            'length_i':length_i,
            'length_pre':length_pre,
            'length_post':length_post,
            'loss':loss,
            'train':train,
        }
        return opt

