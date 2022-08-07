import six
import copy
import math
import random
import numpy as np
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import transformer_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import metrics
from tensor2tensor.models import transformer
import tensorflow as tf
from tensorflow.python.ops import inplace_ops
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
from tensorflow.python.framework import ops
def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm_python(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer = regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def layer_norm_tf(input_tensor, scope=None, reuse=None):
    return tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=scope, reuse=reuse)

norm_fn = layer_norm_python




@registry.register_hparams
def collaboration_tiny():
  hparams = transformer.transformer_tiny()
  hparams.num_hidden_layers = 6
  hparams.add_hparam("reuse_n", 1)
  hparams.add_hparam("attn_kernel_size", 1)
  hparams.add_hparam("reuse_enc", True)
  hparams.add_hparam("reuse_dec", True)
  hparams.add_hparam("reuse_encdec", True)
  hparams.add_hparam("usedecay", 0.8)
  hparams.add_hparam("useweight", False)
  hparams.add_hparam("modelname", "CollFormer")
  return hparams

@registry.register_hparamsE
def collaboration_base():
  hparams = transformer.transformer_base()
  hparams.add_hparam("reuse_n", 1)
  hparams.add_hparam("attn_kernel_size", 1)
  hparams.add_hparam("reuse_enc", True)
  hparams.add_hparam("reuse_dec", True)
  hparams.add_hparam("reuse_encdec", True)
  hparams.add_hparam("usedecay", 0.8)
  hparams.add_hparam("useweight", False)
  hparams.add_hparam("modelname", "CollFormer")
  return hparams

@registry.register_hparams
def collaboration_big():
  hparams = transformer.transformer_big()
  hparams.add_hparam("reuse_n", 1)
  hparams.add_hparam("attn_kernel_size", 1)
  hparams.add_hparam("reuse_enc", True)
  hparams.add_hparam("reuse_dec", True)
  hparams.add_hparam("reuse_encdec", True)
  hparams.add_hparam("usedecay", 0.8)
  hparams.add_hparam("useweight", False)
  hparams.add_hparam("modelname", "CollFormer")
  return hparams


@registry.register_hparams
def collaboration_big_enfr():
  hparams = collaboration_big()
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams

@registry.register_model
class Collaboration(transformer.Transformer):

  def __init__(self, *args, **kwargs):
    super(Collaboration, self).__init__(*args, **kwargs)
    self._encoder_function = _encoder
    self._decoder_function = _decoder

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape    
    
def diff_positions(inputs, name=None):
    """ Calculate the differences of all heads alignment matrices (attention weights)
    :param inputs: A tensor with shape [batch, heads, length_q, length_kv]
    :param name: An optional string
    :returns: A tensor with shape [batch], alignment from sentence to sentence
    """

    with tf.name_scope(name, default_name="diff_positions", values=[inputs]):
        x = inputs
        heads = tf.cast(tf.shape(x)[1], tf.float32)
        x1 = tf.expand_dims(x, 1)  #shape [batch, 1, heads, length_q, length_kv]
        x2 = tf.expand_dims(x, 2)  #shape [batch, heads, 1, length_q, length_kv]

        sos_diff = tf.subtract(x1, x2) #shape [batch, heads, heads, length_q, length_kv], broadcasting
        sos_diff = tf.transpose(sos_diff, [0, 3, 1, 2, 4]) #shape [batch, length_q, heads, heads, length_kv]
        sos_diff = tf.reduce_sum(tf.square(sos_diff), axis=[-3,-2,-1]) / (heads*heads) #shape [batch, length_q]
        # sos_diff_log = tf.negative(tf.log(sos_diff))
        # sos_diff = tf.negative(sos_diff) + 1.0  # Query side needs mask, which is at outside

        mul_diff = tf.multiply(x1, x2) #shape [batch, heads, heads, length_q, length_kv]
        mul_diff = tf.transpose(mul_diff, [0, 3, 1, 2, 4]) #shape [batch, length_q, heads, heads, length_kv]
        mul_diff = tf.reduce_sum(mul_diff, axis=[-3,-2,-1]) / (heads*heads) #shape [batch, length_q]
        # mul_diff_log = tf.negative(tf.log(mul_diff))
        #mul_diff = tf.negative(mul_diff) + 1.0  # Query side needs mask, which is at outside

        cos_diff = tf.multiply(tf.nn.l2_normalize(x1, dim=-1), tf.nn.l2_normalize(x2, dim=-1))
        cos_diff = tf.transpose(cos_diff, [0, 3, 1, 2, 4]) #shape [batch, length_q, heads, heads, length_kv]
        cos_diff = tf.reduce_sum(cos_diff, axis=[-3,-2,-1]) / (heads*heads) #shape [batch, length_q], no need to plus one

        return mul_diff    
    
  
def dot_product_attention(hparams, last_attentions,
                          q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None,
                          activation_dtype=None,
                          weight_dtype=None,
                          hard_attention_k=0,
                          gumbel_noise_weight=0.0):


    if hparams.modelname == 'CollMHA':  
         #J.-B. Cordonnier, A. Loukas, and M. Jaggi, “Multi-head attention:
        # Collaborate instead of concatenate,” arXiv preprint arXiv:2006.16362.,
        # 2020         
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:

          #    logits = tf.einsum("...kd,...qd->...qk", k, q) # batch x num_heads x query_length x memory_length

          # q: [batch, num_heads, query_length, dim]
          # k: [batch, num_heads, memory_length, dim]

          s = tf.shape(q)
          t = tf.shape(k)
          q = tf.reshape(tf.transpose(q, [0,2,1,3]), [s[0], s[2], hparams.hidden_size])
          k = tf.reshape(tf.transpose(k, [0,2,1,3]), [t[0], t[2], hparams.hidden_size])

          mixing = np.zeros([hparams.num_heads, hparams.hidden_size], dtype=np.float32)
          dim_head = int(math.ceil(hparams.hidden_size / hparams.num_heads))
          for i in range(hparams.num_heads):
              mixing[i, i *dim_head: (i + 1) * dim_head] = 1.0

          #         mixing = tf.Variable(mixing, trainable=False, dtype=tf.float32, name="mix")
          mixing = tf.convert_to_tensor(mixing, dtype=tf.float32, name="mix")
                 
          q = tf.expand_dims(q, axis=-3)
          mixing = tf.expand_dims(mixing, axis=-2)
          mixed_q = q*mixing
          mixed_k = tf.expand_dims(k, axis=-3)
          mixed_k = tf.tile(tf.expand_dims(k, axis=-3), [1,hparams.num_heads,1,1])
          
          logits = tf.matmul(mixed_q, mixed_k, transpose_b=True)
          weights = tf.nn.softmax(logits + bias)
          weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

          # if save_weights_to is not None:
          #     save_weights_to[scope.name] = weights
          #     save_weights_to[scope.name + "/logits"] = logits

          return tf.matmul(weights, v), last_attentions
    elif hparams.modelname == 'MHADR':   #disagreement regularization
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:

        #    logits = tf.einsum("...kd,...qd->...qk", k, q) # batch x num_heads x query_length x memory_length
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits + bias)
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

        # if save_weights_to is not None:
        #     save_weights_to[scope.name] = weights
        #     save_weights_to[scope.name + "/logits"] = logits  #

        diff_position = diff_positions(weights)
        loss = tf.reduce_sum(diff_position)
        tf.losses.add_loss(-loss)
        return tf.matmul(weights, v), last_attentions   
    elif hparams.modelname == 'RealFormer':   
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:

          #    logits = tf.einsum("...kd,...qd->...qk", k, q) # batch x num_heads x query_length x memory_length
          logits = tf.matmul(q, k, transpose_b=True)
          if len(last_attentions)>0 and hparams.reuse_n>=0:
              logits = logits+last_attentions[0]
              last_attentions.append(logits)
          weights = tf.nn.softmax(logits + bias)
          weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

          # if save_weights_to is not None:
          #     save_weights_to[scope.name] = weights
          #     save_weights_to[scope.name + "/logits"] = logits

          return tf.matmul(weights, v), last_attentions
    elif hparams.modelname == 'ConcreteGate': 
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:
        logits = tf.matmul(q, k, transpose_b=True)
        

        # s = bias.get_shape().as_list()

        s = get_shape_list(logits)
        logits = logits*(1.0/tf.sqrt(hparams.hidden_size/hparams.num_heads))


        gate_hp={'l0_penalty': 1.0}
        with tf.variable_scope(name,default_name="gate"):
            scope = tf.get_variable_scope()
            gate = ConcreteGate('gate', shape=[1, hparams.num_heads, 1, 1], **gate_hp)
        logits=gate(logits)
        weights = tf.nn.softmax(logits+bias)
      return tf.matmul(weights, v), last_attentions
    elif hparams.modelname == 'MixedMA':     
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:
          logits = tf.matmul(q, k, transpose_b=True)
          # s = bias.get_shape().as_list()
          s = get_shape_list(logits)
          logits = logits*(1.0/tf.sqrt(hparams.hidden_size/hparams.num_heads))

          # s = logits.shape[2]
          bias = bias + tf.zeros_like(logits)
          k=hparams.num_heads//4
          A = tf.linalg.band_part(bias[:,0:k,:,:],1,1)
          B = tf.linalg.band_part(bias[:,k:2*k,:,:],-1,0)
          C = tf.linalg.band_part(bias[:,2*k:3*k,:,:],0,-1)
          D=  bias[:,3*k:4*k,:,:]
          mask = tf.concat([A,B,C,D],axis=1)
          mask = tf.cast(mask<0,dtype=tf.float32)*(-1.0e9)
          
          weights = tf.nn.softmax(logits+mask+bias)
      return tf.matmul(weights, v), last_attentions    
    elif hparams.modelname == 'TalkingHA':   
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:
          logits = tf.matmul(q, k, transpose_b=True)
          logits = tf.transpose(logits, [0, 2, 3, 1])
          logits = tf.layers.Dense(hparams.num_heads, use_bias=False, name="attn_ffn_1")(logits)
          logits = tf.transpose(logits, [0, 3, 1, 2])
          weights = tf.nn.softmax(logits + bias)
          weights = tf.transpose(weights, [0, 2, 3, 1])
          weights = tf.layers.Dense(hparams.num_heads, use_bias=False, name="attn_ffn_2")(weights)
          weights = tf.transpose(weights, [0, 3, 1, 2])
      return tf.matmul(weights, v), last_attentions
    elif hparams.modelname == 'CollFormer': 
      with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:

          #    logits = tf.einsum("...kd,...qd->...qk", k, q) # batch x num_heads x query_length x memory_length
          logits = tf.matmul(q, k, transpose_b=True)
          logits0 = logits*(1.0/tf.sqrt(hparams.hidden_size/hparams.num_heads))

          with tf.variable_scope(name, default_name="collvars"):
              if last_attentions is not None and hparams.reuse_n>0:

                  last_attn = last_attentions[-hparams.reuse_n:]
                  if hparams.useweight:
                      layer_weights = tf.get_variable("layer_weights", shape=[len(last_attn)], initializer=tf.zeros_initializer())
                      layer_weights = tf.nn.softmax(layer_weights, dim=0)
                      last_attn = [last_attn[i] * layer_weights[i] for i in range(len(last_attn))]

                  if hparams.usedecay > 0:
                      last_attn = [last_attn[-i] * (hparams.usedecay ** i) for i in
                                  range(1, min(hparams.reuse_n, len(last_attn)) + 1)]

            
                  channels = hparams.num_heads * (1 + len(last_attn))
                  
              
                  x = tf.concat([logits0] + last_attn, axis=1)
                  x = tf.transpose(x, [0, 2, 3, 1])
                  x.set_shape(x.shape.as_list()[:-1] + [channels])
                  if hparams.attn_kernel_size == 1:
                      # x = tf.contrib.layers.layer_norm(x)
                      x = tf.nn.dropout(x,1.0 - dropout_rate)
                      x = tf.layers.Dense(channels, use_bias=True, name="attn_ffn_1")(x)
                      x = tf.nn.relu(x)
                      x = tf.layers.Dense(hparams.num_heads, use_bias=True, name="attn_ffn_2")(x)
                  else:
                      x = conv(x, channels, bias=True, activation=tf.nn.relu, kernel_size=hparams.attn_kernel_size,
                              scope="attn_conv_1")
                      x = conv(x, hparams.num_heads, bias=True, kernel_size=hparams.attn_kernel_size, scope="attn_conv_2")
                  
                  # x = tf.contrib.layers.layer_norm(x,begin_norm_axis=2,begin_params_axis=-1,scope='layer_norm')   #%%%%%%%%%layer_norm
                      
                  logits = tf.transpose(x, [0, 3, 1, 2])
                  last_attentions.append(logits)
              elif hparams.reuse_n==0:
                  x = logits0
                  channels = hparams.num_heads
                  x = tf.transpose(x, [0, 2, 3, 1])
                  x = tf.nn.dropout(x,1.0 - dropout_rate)
                  x = tf.layers.Dense(channels, use_bias=True, name="attn_ffn_1")(x)
                  x = tf.nn.relu(x)
                  x = tf.layers.Dense(hparams.num_heads, use_bias=True, name="attn_ffn_2")(x) 
                  # x = tf.contrib.layers.layer_norm(x,begin_norm_axis=2,begin_params_axis=-1, scope='layer_norm')               #%%%%%%%%%layer_norm            
                  
                  logits = tf.transpose(x, [0, 3, 1, 2])              
                  last_attentions.append(logits)


          if  hparams.only_collvars:
              only_collvars=tf.constant(1)  
          else:
              only_collvars=tf.constant(0)  
          
          logits = tf.cond(only_collvars>tf.constant(0),lambda:logits, lambda:logits0)

          weights = tf.nn.softmax(logits + bias)
          weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

          # if save_weights_to is not None:
          #     save_weights_to[scope.name] = weights
          #     save_weights_to[scope.name + "/logits"] = logits

          return tf.matmul(weights, v), last_attentions



    
def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, scope="conv", reuse=None, padding="SAME", bias_initializer=tf.zeros_initializer):
  with tf.variable_scope(scope, reuse = reuse):
    shapes = inputs.shape.as_list()
    if len(shapes) > 4:
      raise NotImplementedError
    elif len(shapes) == 4:
      filter_shape = [1,kernel_size,shapes[-1],output_size]
      bias_shape = [1,1,1,output_size]
      strides = [1,1,1,1]
    else:
      filter_shape = [kernel_size,shapes[-1],output_size]
      bias_shape = [1,1,output_size]
      strides = 1
    conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
    kernel_ = tf.get_variable("kernel_", filter_shape, dtype = tf.float32)
    outputs = conv_func(inputs, kernel_, strides, padding)
    if bias:
      bias_ = tf.get_variable('bias_', bias_shape, dtype=tf.float32, initializer=bias_initializer())
      outputs = outputs + bias_
    if activation is not None:
      return activation(outputs)
    return outputs


###################################################################################
#  The code below is copied from https://github.com/lena-voita/the-story-of-heads,#
#  used for ConcreteGate                                                          #
###################################################################################
import hashlib
from copy import copy


def get_seed_from_name(name):
    full_name = '/'.join([tf.get_variable_scope().name, name])
    return int(hashlib.md5(full_name.encode()).hexdigest()[:8], 16)


def default_initializer(seed, dtype):
    scope_initializer = tf.get_variable_scope().initializer
    if scope_initializer is not None:
        return scope_initializer
    try:
        return tf.initializers.glorot_uniform(seed, dtype)
    except:
        return tf.glorot_uniform_initializer(seed, dtype)


def get_model_variable(name, **kwargs):
    """ Get variable from MODEL_VARIABLES collection with initializer seeded from its name, not id """

    if kwargs.get('initializer') is None:
        kwargs['initializer'] = default_initializer(seed=get_seed_from_name(name), dtype=kwargs.get('dtype', tf.float32))
    elif hasattr(kwargs['initializer'], 'seed') and kwargs['initializer'].seed is None:
        kwargs['initializer'] = copy(kwargs['initializer'])
        kwargs['initializer'].seed = get_seed_from_name(name)

    return tf.contrib.framework.model_variable(name, **kwargs)

class ConcreteGate:
    """
    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Example usage: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in https://arxiv.org/pdf/1712.01312.pdf
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param l2_penalty: coefficient on the regularizer that minimizes l2 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    :param hard: if True, gates are binarized to {0, 1} but backprop is still performed as if they were concrete
    :param local_rep: if True, samples a different gumbel noise tensor for each sample in batch,
        by default, noise is sampled using shape param as size.
    """

    def __init__(self, name, shape, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=0.0, l2_penalty=0.0, eps=1e-6, hard=False, local_rep=False):
        self.name = name
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.l0_penalty, self.l2_penalty = l0_penalty, l2_penalty
        self.hard, self.local_rep = hard, local_rep
        with tf.variable_scope(name):
            self.log_a = get_model_variable("log_a", shape=shape)

    def __call__(self, values, is_train=None, axis=None, reg_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        gates = self.get_gates(is_train, shape=tf.shape(values) if self.local_rep else None)

        if self.l0_penalty != 0 or self.l2_penalty != 0:
            reg = self.get_penalty(values=values, axis=axis)
            tf.add_to_collection(reg_collection, tf.identity(reg, name='concrete_gate_reg'))
        return values * gates

    def get_gates(self, is_train, shape=None):
        """ samples gate activations in [0, 1] interval """
        low, high = self.stretch_limits
        with tf.name_scope(self.name):
            if is_train:
                shape = tf.shape(self.log_a) if shape is None else shape
                noise = tf.random_uniform(shape, self.eps, 1.0 - self.eps)
                concrete = tf.nn.sigmoid((tf.log(noise) - tf.log(1 - noise) + self.log_a) / self.temperature)
            else:
                concrete = tf.nn.sigmoid(self.log_a)

            stretched_concrete = concrete * (high - low) + low
            clipped_concrete = tf.clip_by_value(stretched_concrete, 0, 1)
            if self.hard:
                hard_concrete = tf.to_float(tf.greater(clipped_concrete, 0.5))
                clipped_concrete = clipped_concrete + tf.stop_gradient(hard_concrete - clipped_concrete)
        return clipped_concrete


    def get_penalty(self, values=None, axis=None):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        with tf.name_scope(self.name):
            # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
            p_open = tf.nn.sigmoid(self.log_a - self.temperature * tf.log(-low / high))
            p_open = tf.clip_by_value(p_open, self.eps, 1.0 - self.eps)

            total_reg = 0.0
            if self.l0_penalty != 0:
                if values != None and self.local_rep:
                    p_open += tf.zeros_like(values)  # broadcast shape to account for values
                l0_reg = self.l0_penalty * tf.reduce_sum(p_open, axis=axis)
                total_reg += tf.reduce_mean(l0_reg)

            if self.l2_penalty != 0:
                assert values is not None
                l2_reg = 0.5 * self.l2_penalty * p_open * tf.reduce_sum(values ** 2, axis=axis)
                total_reg += tf.reduce_mean(l2_reg)

            return total_reg

    def get_sparsity_rate(self, is_train=False):
        """ Computes the fraction of gates which are now active (non-zero) """
        is_nonzero = tf.not_equal(self.get_gates(is_train), 0.0)
        return tf.reduce_mean(tf.to_float(is_nonzero))



###################################################################################
#  The code below is copied from tensor2tensor(v1.14.1) with minor modifications  #
###################################################################################

def _encoder(encoder_input,
             encoder_self_attention_bias,
             hparams,
             name="encoder",
             nonpadding=None,
             save_weights_to=None,
             make_image_summary=True,
             losses=None,
             attn_bias_for_padding=None):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convolutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    attn_bias_for_padding: Padded attention bias in case a unidirectional
      encoder is being used where future attention is masked.

  Returns:
    y: a Tensors
  """
  x = encoder_input

  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        attention_bias = attn_bias_for_padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)
    last_attentions = [] if hparams.reuse_enc else None
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          if layer < hparams.get("num_area_layers", 0):
            max_area_width = hparams.get("max_area_width", 1)
            max_area_height = hparams.get("max_area_height", 1)
            memory_height = hparams.get("memory_height", 1)
          else:
            max_area_width = 1
            max_area_height = 1
            memory_height = 1
          y, last_attentions = multihead_attention(
              hparams,
              last_attentions,
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              hard_attention_k=hparams.get("hard_attention_k", 0),
              gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
              max_area_width=max_area_width,
              max_area_height=max_area_height,
              memory_height=memory_height,
              area_key_mode=hparams.get("area_key_mode", "none"),
              area_value_mode=hparams.get("area_value_mode", "none"),
              training=(hparams.get("mode", tf.estimator.ModeKeys.TRAIN)
                        == tf.estimator.ModeKeys.TRAIN))
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_layers.transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              pad_remover,
              conv_padding="SAME",
              nonpadding_mask=nonpadding,
              losses=losses)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)

def _decoder(decoder_input,
             encoder_output,
             decoder_self_attention_bias,
             encoder_decoder_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             name="decoder",
             nonpadding=None,
             save_weights_to=None,
             make_image_summary=True,
             losses=None,
             layer_collection=None,
             recurrent_memory_by_layer=None,
             chunk_number=None):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.

  Returns:
    y: a Tensors
  """
  x = decoder_input

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      },
      hparams=hparams)

  with tf.variable_scope(name):
    last_attentions_dec = [] if hparams.reuse_dec else None
    last_attentions_encdec = [] if hparams.reuse_encdec else None
    for layer_idx in range(hparams.num_decoder_layers or
                           hparams.num_hidden_layers):
      x, last_attentions_dec, last_attentions_encdec = transformer_decoder_layer(
          x, last_attentions_dec, last_attentions_encdec,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          encoder_output=encoder_output,
          cache=cache,
          decode_loop_step=decode_loop_step,
          nonpadding=nonpadding,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          losses=losses,
          layer_collection=layer_collection,
          recurrent_memory_by_layer=recurrent_memory_by_layer,
          chunk_number=chunk_number
          )

    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(
        x, hparams, layer_collection=layer_collection)

def transformer_decoder_layer(decoder_input, last_attentions_dec, last_attentions_encdec,
                              decoder_self_attention_bias,
                              layer_idx,
                              hparams,
                              encoder_output=None,
                              encoder_decoder_attention_bias=None,
                              cache=None,
                              decode_loop_step=None,
                              nonpadding=None,
                              save_weights_to=None,
                              make_image_summary=False,
                              losses=None,
                              layer_collection=None,
                              recurrent_memory_by_layer=None,
                              chunk_number=None):
  """A single transformer decoder layer."""
  x, layer_cache, last_attentions_dec, last_attentions_encdec = transformer_self_attention_layer(
      last_attentions_dec, last_attentions_encdec,
      decoder_input=decoder_input,
      decoder_self_attention_bias=decoder_self_attention_bias,
      layer_idx=layer_idx,
      hparams=hparams,
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      cache=cache,
      decode_loop_step=decode_loop_step,
      save_weights_to=save_weights_to,
      make_image_summary=make_image_summary,
      layer_collection=layer_collection,
      recurrent_memory_by_layer=recurrent_memory_by_layer,
      chunk_number=chunk_number)

  layer = layer_idx
  layer_name = "layer_%d" % layer
  if hparams.filter_size > 0:
    with tf.variable_scope(layer_name):
      with tf.variable_scope("ffn"):
        y = transformer_layers.transformer_ffn_layer(
            common_layers.layer_preprocess(
                x, hparams, layer_collection=layer_collection),
            hparams,
            conv_padding="LEFT",
            nonpadding_mask=nonpadding,
            losses=losses,
            cache=layer_cache,
            decode_loop_step=decode_loop_step,
            layer_collection=layer_collection)
        x = common_layers.layer_postprocess(x, y, hparams)
  return x, last_attentions_dec, last_attentions_encdec

def transformer_self_attention_layer(last_attentions_dec, last_attentions_encdec,
                                     decoder_input,
                                     decoder_self_attention_bias,
                                     layer_idx,
                                     hparams,
                                     encoder_output=None,
                                     encoder_decoder_attention_bias=None,
                                     cache=None,
                                     decode_loop_step=None,
                                     save_weights_to=None,
                                     make_image_summary=False,
                                     layer_collection=None,
                                     recurrent_memory_by_layer=None,
                                     chunk_number=None):
  """A single transformer self-attention layer."""
  x = decoder_input
  layer = layer_idx
  layer_name = "layer_%d" % layer
  layer_cache = cache[layer_name] if cache is not None else None

  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  if recurrent_memory_by_layer is not None:
    recurrent_memory = recurrent_memory_by_layer[layer_name]
  else:
    recurrent_memory = None

  if layer < hparams.get("num_area_layers", 0):
    max_area_width = hparams.get("max_area_width", 1)
    max_area_height = hparams.get("max_area_height", 1)
    memory_height = hparams.get("max_area_height", 1)
  else:
    max_area_width = 1
    max_area_height = 1
    memory_height = 1
  with tf.variable_scope(layer_name):
    with tf.variable_scope("self_attention"):
      y, last_attentions_dec = multihead_attention(
          hparams,
          last_attentions_dec,
          common_layers.layer_preprocess(
              x, hparams, layer_collection=layer_collection),
          None,
          decoder_self_attention_bias,
          hparams.attention_key_channels or hparams.hidden_size,
          hparams.attention_value_channels or hparams.hidden_size,
          hparams.hidden_size,
          hparams.num_heads,
          hparams.attention_dropout,
          attention_type=hparams.self_attention_type,
          max_relative_position=hparams.max_relative_position,
          heads_share_relative_embedding=(
              hparams.heads_share_relative_embedding),
          add_relative_to_values=hparams.add_relative_to_values,
          save_weights_to=save_weights_to,
          cache=layer_cache,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=attention_dropout_broadcast_dims,
          max_length=hparams.get("max_length"),
          decode_loop_step=decode_loop_step,
          vars_3d=hparams.get("attention_variables_3d"),
          activation_dtype=hparams.get("activation_dtype", "float32"),
          weight_dtype=hparams.get("weight_dtype", "float32"),
          layer_collection=layer_collection,
          recurrent_memory=recurrent_memory,
          chunk_number=chunk_number,
          hard_attention_k=hparams.get("hard_attention_k", 0),
          gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
          max_area_width=max_area_width,
          max_area_height=max_area_height,
          memory_height=memory_height,
          area_key_mode=hparams.get("area_key_mode", "none"),
          area_value_mode=hparams.get("area_value_mode", "none"),
          training=(hparams.get(
              "mode",
              tf.estimator.ModeKeys.TRAIN) == tf.estimator.ModeKeys.TRAIN))
      x = common_layers.layer_postprocess(x, y, hparams)
    if encoder_output is not None:
      with tf.variable_scope("encdec_attention"):
        y, last_attentions_encdec = multihead_attention(
            hparams,
            last_attentions_encdec,
            common_layers.layer_preprocess(
                x, hparams, layer_collection=layer_collection),
            encoder_output,
            encoder_decoder_attention_bias,
            hparams.attention_key_channels or hparams.hidden_size,
            hparams.attention_value_channels or hparams.hidden_size,
            hparams.hidden_size,
            hparams.num_heads,
            hparams.attention_dropout,
            max_relative_position=hparams.max_relative_position,
            heads_share_relative_embedding=(
                hparams.heads_share_relative_embedding),
            add_relative_to_values=hparams.add_relative_to_values,
            save_weights_to=save_weights_to,
            cache=layer_cache,
            make_image_summary=make_image_summary,
            dropout_broadcast_dims=attention_dropout_broadcast_dims,
            max_length=hparams.get("max_length"),
            vars_3d=hparams.get("attention_variables_3d"),
            activation_dtype=hparams.get("activation_dtype", "float32"),
            weight_dtype=hparams.get("weight_dtype", "float32"),
            layer_collection=layer_collection,
            hard_attention_k=hparams.get("hard_attention_k", 0),
            gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
            max_area_width=max_area_width,
            max_area_height=max_area_height,
            memory_height=memory_height,
            area_key_mode=hparams.get("area_key_mode", "none"),
            area_value_mode=hparams.get("area_value_mode", "none"),
            training=(hparams.get(
                "mode",
                tf.estimator.ModeKeys.TRAIN) == tf.estimator.ModeKeys.TRAIN))
        x = common_layers.layer_postprocess(x, y, hparams)
    return x, layer_cache, last_attentions_dec, last_attentions_encdec

def multihead_attention(hparams,
                         last_attentions,
                         query_antecedent,
                         memory_antecedent,
                         bias,
                         total_key_depth,
                         total_value_depth,
                         output_depth,
                         num_heads,
                         dropout_rate,
                         attention_type="dot_product",
                         max_relative_position=None,
                         heads_share_relative_embedding=False,
                         add_relative_to_values=False,
                         image_shapes=None,
                         block_length=128,
                         block_width=128,
                         q_filter_width=1,
                         kv_filter_width=1,
                         q_padding="VALID",
                         kv_padding="VALID",
                         cache=None,
                         gap_size=0,
                         num_memory_blocks=2,
                         name="multihead_attention",
                         save_weights_to=None,
                         make_image_summary=True,
                         dropout_broadcast_dims=None,
                         vars_3d=False,
                         layer_collection=None,
                         recurrent_memory=None,
                         chunk_number=None,
                         hard_attention_k=0,
                         gumbel_noise_weight=0.0,
                         max_area_width=1,
                         max_area_height=1,
                         memory_height=1,
                         area_key_mode="mean",
                         area_value_mode="sum",
                         training=True,
                         **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    heads_share_relative_embedding: boolean to share relative embeddings
    add_relative_to_values: a boolean for whether to add relative component to
                            values.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory: An optional transformer_memory.RecurrentMemory, which
      retains state across chunks. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    training: indicating if it is in the training mode.
    **kwargs (dict): Parameters for the attention function.

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  assert vars_3d == False
  vars_3d_num_heads = num_heads if vars_3d else 0

  assert layer_collection is None
  if layer_collection is not None:
    if cache is not None:
      raise ValueError("KFAC implementation only supports cache is None.")
    if vars_3d:
      raise ValueError("KFAC implementation does not support 3d vars.")

  assert recurrent_memory is None
  if recurrent_memory is not None:
    if memory_antecedent is not None:
      raise ValueError("Recurrent memory requires memory_antecedent is None.")
    if cache is not None:
      raise ValueError("Cache is not supported when using recurrent memory.")
    if vars_3d:
      raise ValueError("3d vars are not supported when using recurrent memory.")
    if layer_collection is not None:
      raise ValueError("KFAC is not supported when using recurrent memory.")
    if chunk_number is None:
      raise ValueError("chunk_number is required when using recurrent memory.")

  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    if recurrent_memory is not None:
      (
          recurrent_memory_transaction,
          query_antecedent, memory_antecedent, bias,
      ) = recurrent_memory.pre_attention(
          chunk_number,
          query_antecedent, memory_antecedent, bias,
      )

    if cache is None or memory_antecedent is None:
      q, k, v = common_attention.compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads,
                            layer_collection=layer_collection)
    if cache is not None:
      if attention_type not in ["dot_product", "dot_product_relative"]:
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = common_attention.compute_attention_component(query_antecedent, total_key_depth,
                                            q_filter_width, q_padding, "q",
                                            vars_3d_num_heads=vars_3d_num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = common_attention.split_heads(k, num_heads)
        v = common_attention.split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

    q = common_attention.split_heads(q, num_heads)
    if cache is None:
      k = common_attention.split_heads(k, num_heads)
      v = common_attention.split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head ** -0.5

    assert attention_type == "dot_product"
    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      if max_area_width > 1 or max_area_height > 1:
        assert 0
        x = area_attention.dot_product_area_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            dropout_broadcast_dims=dropout_broadcast_dims,
            max_area_width=max_area_width,
            max_area_height=max_area_height,
            memory_height=memory_height,
            area_key_mode=area_key_mode,
            area_value_mode=area_value_mode,
            training=training)
      else:
        x, last_attentions = dot_product_attention(hparams, last_attentions,
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            make_image_summary=make_image_summary,
            dropout_broadcast_dims=dropout_broadcast_dims,
            activation_dtype=kwargs.get("activation_dtype"),
            hard_attention_k=hard_attention_k,
            gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_relative":
      x = dot_product_attention_relative(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          cache=cache is not None,
          allow_memory=recurrent_memory is not None,
          hard_attention_k=hard_attention_k,
          gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_unmasked_relative_v2":
      x = dot_product_unmasked_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "dot_product_relative_v2":
      x = dot_product_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "local_within_block_mask_right":
      x = masked_within_block_local_attention_1d(
          q, k, v, block_length=block_length)
    elif attention_type == "local_relative_mask_right":
      x = masked_relative_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary,
          dropout_rate=dropout_rate,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values,
          name="masked_relative_local_attention_1d")
    elif attention_type == "local_mask_right":
      x = masked_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary)
    elif attention_type == "local_unmasked":
      x = local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                           gap_size, num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = dilated_self_attention_1d(q, k, v, block_length, block_width,
                                    gap_size, num_memory_blocks)
    x = common_attention.combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, name="output_transform",
          layer_collection=layer_collection)

    if recurrent_memory is not None:
      x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x, last_attentions

