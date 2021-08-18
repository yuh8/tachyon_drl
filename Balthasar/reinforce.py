import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.embed_utils import import_tf_model
from src.data_gen_utils import generate_smile
from src.reward_utils import get_terminal_reward, get_padded_reward_vec, is_valid_smile
from src.CONSTS import MOL_DICT, ALPHA, BATCH_SIZE, MAX_MOL_LEN


def loss_function(distributed_reward_with_idx, action_probs):
    # [BATCH, MAX_MOL_LEN]
    distributed_reward = distributed_reward_with_idx[:, 0, :]
    # [BATCH, MAX_MOL_LEN, 1]
    distributed_reward = tf.expand_dims(distributed_reward, axis=-1)
    # [BATCH, MAX_MOL_LEN]
    reward_idx = tf.cast(distributed_reward_with_idx[:, 1, :], tf.int32)
    # [BATCH, MAX_MOL_LEN, MOL_DICT_LEN + 1]
    reward_onehot = tf.one_hot(reward_idx, len(MOL_DICT) + 1,
                               on_value=1, off_value=0, axis=-1)
    reward_onehot = tf.cast(reward_onehot, distributed_reward.dtype)
    distributed_reward = distributed_reward * reward_onehot
    # [BATCH, MAX_MOL_LEN]
    mask = tf.math.less(reward_idx, len(MOL_DICT))
    # [BATCH, MAX_MOL_LEN, 1]
    mask = tf.expand_dims(mask, axis=-1)
    # [BATCH, MAX_MOL_LEN, MOL_DICT_LEN + 1]
    loss_ = -action_probs * distributed_reward + ALPHA * tf.math.log(action_probs)
    mask = tf.cast(mask, loss_.dtype)
    loss_ *= mask
    # [BATCH, MAX_MOL_LEN]
    loss_ = tf.reduce_sum(loss_, axis=-1)
    # [BATCH]
    loss_ = tf.reduce_sum(loss_, axis=-1)
    # Scalar
    loss_ = tf.reduce_mean(loss_)
    return loss_


def get_optimizer(steps_per_epoch):
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [steps_per_epoch * 10, steps_per_epoch * 20], [0.00001, 0.000001, 0.0000001], name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    return opt_op


def get_policy_model(caspar):
    smi_inputs = layers.Input(shape=(MAX_MOL_LEN,), dtype=np.int32)
    # [BATCH, MAX_MOL_LEN, DICT_LEN + 1]
    logits = caspar(smi_inputs)
    action_probs = tf.nn.softmax(logits, axis=-1)
    caspar_rl = keras.Model(smi_inputs, action_probs)
    caspar_rl.compile(optimizer=get_optimizer(1000), loss=loss_function)
    return caspar_rl


def caspar_loss_func(y, logits):
    '''
    y : [BATCH, MAX_MOL_LEN]
    logits: [BATCH, MAX_MOL_LEN, DICT_SIZE]
    '''
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.less(y, len(MOL_DICT))
    _loss = loss_obj(y, logits)
    mask = tf.cast(mask, _loss.dtype)
    _loss *= mask
    return tf.reduce_sum(_loss) / tf.reduce_sum(mask)


def load_pretrain_caspar():
    caspar = import_tf_model("pretrained_caspar_model/", caspar_loss_func)
    return caspar


if __name__ == "__main__":
    caspar = load_pretrain_caspar()
    caspar.trainable = True
    caspar_rl = get_policy_model(caspar)
    melchior = import_tf_model("melchior_model/")
    smi_bank = []
    running_reward = 0
    running_validity = 0
    for ii in range(10000):
        input_batch = []
        target_batch = []
        batch_reward = 0
        batch_validity = 0
        for _ in range(BATCH_SIZE):
            start_tokens, start_tokens_ids, generated_tokens, generated_token_ids = generate_smile(caspar_rl)
            if generated_tokens[-1] == "E":
                smi = "".join(generated_tokens[:-1])
            else:
                smi = "".join(generated_tokens)

            if is_valid_smile(smi):
                batch_validity += 1
                if len(smi_bank) >= 20:
                    smi_bank = smi_bank[:-1]
                    smi_bank.insert(0, smi)
                else:
                    smi_bank.append(smi)

            r, T = get_terminal_reward(generated_tokens, smi_bank, melchior)
            r_vec = get_padded_reward_vec(r, 0.9, T)
            distributed_reward_with_idx = np.vstack([r_vec, np.array(generated_token_ids)])
            input_batch.append(generated_token_ids)
            target_batch.append(distributed_reward_with_idx)
            batch_reward += r
        running_reward = 0.05 * batch_reward + (1 - 0.05) * running_reward
        running_validity = 0.05 * (batch_validity / BATCH_SIZE) + (1 - 0.05) * running_validity
        input_batch = np.vstack(input_batch)
        target_batch = np.stack(target_batch)
        loss = caspar_rl.train_on_batch(input_batch, target_batch)
        print("train_loss={0}, reward={1} and validity={2} at iteration {3}".format(np.round(loss, 2),
                                                                                    np.round(running_reward, 2),
                                                                                    np.round(running_validity, 2), ii))
        if ii % 100 == 0:
            caspar_rl.save_weights('./checkpoints/caspar_rl')
            caspar_rl.save('model/Balthasar/', save_traces=False)
