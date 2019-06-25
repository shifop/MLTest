from kw.keywords import *
from util.tools import read_json,write_json


if __name__=='__main__':
    config = TCNNConfig()
    config.test_data_path = '../data/神箭关键词提取/test2.record'
    config.batch_size = 1
    oj = TextCNN(config)
    data = read_json('../data/神箭关键词提取/test2.json')
    ids = read_json('../data/神箭关键词提取/docs.json')
    path = '../model/20190602202700/model.ckpt'
    rt=[]
    with tf.Session(graph=oj.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        oj.saver.restore(sess, path)
        sess.run(oj.test_data_op)
        for index,x in enumerate(tqdm(data)):
            p = sess.run(oj.test_pt)[0]
            p_o = [x[0] for x in p]
            p = list(p_o)
            p_s = sorted(p,reverse=True)[:2]
            p_index = [p_o.index(i) for i in p_s]
            rt.append({'id':ids[index]['id'],'keywords':[x['keywords'][i] for i in p_index]})

    write_json(rt,'../data/神箭关键词提取/rt2.json')
