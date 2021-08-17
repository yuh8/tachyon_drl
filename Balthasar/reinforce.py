import numpy as np
import tensorflow as tf
from src.embed_utils import import_tf_model
from src.data_gen_utils import generate_smile
from src.reward_utils import get_terminal_reward, get_padded_reward_vec
from src.CONSTS import MOL_DICT, ALPHA, BATCH_SIZE


def loss_function(action_prob, distributed_reward_with_idx):
    # [BATCH, MAX_MOL_LEN]
    distributed_reward = distributed_reward_with_idx[:, 0, :]
    # [BATCH, MAX_MOL_LEN, 1]
    distributed_reward = tf.expand_dims(distributed_reward, axis=-1)
    # [BATCH, MAX_MOL_LEN]
    reward_idx = distributed_reward_with_idx[:, 1, :]
    # [BATCH, MAX_MOL_LEN, MOL_DICT_LEN + 1]
    reward_onehot = tf.one_hot(reward_idx, len(MOL_DICT) + 1,
                               on_value=1, off_value=0, axis=-1)
    distributed_reward = distributed_reward * reward_onehot
    # [BATCH, MAX_MOL_LEN]
    mask = tf.math.less(reward_idx, len(MOL_DICT))
    # [BATCH, MAX_MOL_LEN, 1]
    mask = tf.expand_dims(mask, axis=-1)
    # [BATCH, MAX_MOL_LEN, MOL_DICT_LEN + 1]
    loss_ = -action_prob * distributed_reward + ALPHA * tf.math.log(action_prob)
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
        [steps_per_epoch * 10, steps_per_epoch * 20], [0.0001, 0.00001, 0.000001], name=None
    )
    opt_op = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    return opt_op


def compile_policy_model(caspar):
    caspar.compile(optimizer=get_optimizer(1000), loss=loss_function)
    return caspar


if __name__ == "__main__":
    caspar = import_tf_model("pretrained_caspar_model")
    caspar_rl = compile_policy_model(caspar)
    melchior = import_tf_model("melchior_model")
    smi_bank = []
    running_reward = 0
    for ii in range(10000):
        input_batch = []
        target_batch = []
        batch_reward = 0
        for _ in range(BATCH_SIZE):
            generated_tokens, generated_token_ids = generate_smile(caspar_rl)
            smi = "".join(generated_tokens)
            if len(smi_bank) >= 20:
                smi_bank = smi_bank.pop(-1)
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
        input_batch = np.array(input_batch)
        target_batch = np.stack(target_batch)
        loss = caspar_rl.train_on_batch(input_batch, target_batch)
        print("train_loss={0}, reward={1} at iteration {2}".format(loss, running_reward, ii))
        if ii % 100 == 0:
            caspar_rl.save_weights('./checkpoints/caspar_rl')
            caspar_rl.save('model/Balthasar/', save_traces=False)
