

# a = range(10)
# b = range(11,20)
# c = []
# for i in range(10):
#     c.append((i*2,i*3))
# k = zip(a,b)
# for i in k:
#     print(i)
# for i in zip(*c):
#     print(list(i))
#
import tensorflow as tf

def t1():
    with tf.Session() as sess:
        seql = range(5)
        seqv = [i*5 for i in range(5)]

        state_ta = tf.TensorArray(dtype=tf.float32,size=100)

        mat_data = [range(5) for i in range(4)]
        length = range(1,5)
        i = tf.constant(0)
        tmp = tf.get_variable('k',shape=[5,2],dtype=tf.float32)
        def _encoder_evid(i,state_ta):
            vec = tf.constant(1.0,shape=[10])
            # vec = tf.pad(vec,[[0,6-evid_len[i]],[0,0]])
            # vec = tf.reshape(vec,[6,2])
            state_ta = state_ta.write(i,vec)
            print('123')
            i = tf.add(i,1)
            return i,state_ta
        loop, state_ta = tf.while_loop(lambda i,state_ta: i < 5, _encoder_evid,[i,state_ta])

        init = tf.global_variables_initializer()
        sess.run(init)
        with tf.control_dependencies([loop]):
            ta_t = state_ta.stack()
        i,r = sess.run([loop,ta_t],feed_dict={})
        print(i)
        print(r)

t1()
