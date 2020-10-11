import tensorflow as tf
from absl import app, flags
 
# 定义输入参数
FLAGS = flags.FLAGS
flags.DEFINE_string('pb_name', None, 'the pb file freezen')
flags.DEFINE_string('tflite_name', None, 'the tflite name')
flags.DEFINE_integer('input_size', 608, 'the input image size for tf')

###################################################
def convert(in_pb_path, out_tflite_path, image_input_size):
    # in_pb_path = r'.\yolov3_coco.pb'
    # out_tflite_path = r'.\yolov3_coco.tflite'
     
    input_arrays = ["input/input_data"]
    input_shapes = {"input/input_data" :[1, image_input_size, image_input_size, 3]}
    output_arrays = ["pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
     
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(in_pb_path, input_arrays, output_arrays, input_shapes)
    tflite_model = converter.convert()
    open(out_tflite_path, "wb").write(tflite_model)

###################################################
def main(argv):
    del argv
    if not FLAGS.pb_name:
        print('[ERROR] please set pb_name')
        raise()
    
    if not FLAGS.tflite_name:
        print('[ERROR] please set tflite_name')
        raise()
    
    convert(FLAGS.pb_name, FLAGS.tflite_name, FLAGS.input_size)

#################################################
if __name__ == '__main__':
    app.run(main)
